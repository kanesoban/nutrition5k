import time
import logging
import os
from glob import glob
from shutil import copyfile
import datetime as dt

'''
DEBUG_MODE == True is intended for testing code, while 
DEBUG_MODE == False is intended to enable faster training and inference
'''
DEBUG_MODE = True
import numpy as np
import torch

CUDA_VERSION = torch.version.cuda
logging.warning('cuda version: {}'.format(CUDA_VERSION))
'''
if CUDA_VERSION:
    logging.warning('CUDA_PATH: {}'.format(os.environ['CUDA_PATH']))
    logging.warning('CUDA_HOME: {}'.format(os.environ['CUDA_HOME']))
'''

if DEBUG_MODE:
    np.random.seed(0)
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    torch.autograd.profiler.profile(True)
    torch.backends.cudnn.benchmark = False
else:
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_printoptions(linewidth=120)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from nutrition5k import n5kloss
from nutrition5k.dataset import Resize, ToTensor, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, Normalize, \
    Nutrition5kDataset, create_nutrition_df, split_dataframe
from nutrition5k.model import Nutrition5kModel
from nutrition5k.train_utils import run_epoch
from nutrition5k.utils import parse_args, Metrics

SECONDS_TO_HOURS = 3600
IMAGE_RESOUTION = (299, 299)


def create_dataloaders():
    nutrition_df = create_nutrition_df(config['dataset_dir'])
    train_df, val_df, test_df = split_dataframe(nutrition_df, config['split'])

    train_set = Nutrition5kDataset(train_df, config['dataset_dir'], transform=transforms.Compose(
        [Resize(IMAGE_RESOUTION),
         ToTensor(),
         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    val_set = Nutrition5kDataset(val_df, config['dataset_dir'], transform=transforms.Compose(
        [Resize(IMAGE_RESOUTION),
         ToTensor(),
         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    test_set = Nutrition5kDataset(test_df, config['dataset_dir'], transform=transforms.Compose(
        [Resize(IMAGE_RESOUTION),
         ToTensor(),
         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    return {
        'train': DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,
                            num_workers=config['dataset_workers'], pin_memory=True),
        'val': DataLoader(val_set, batch_size=config['batch_size'], shuffle=False,
                          num_workers=config['dataset_workers'], pin_memory=True),
        'test': DataLoader(test_set, batch_size=config['batch_size'], shuffle=False,
                           num_workers=config['dataset_workers'], pin_memory=True)
    }


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    with open(args.config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    comment = f' batch_size = {config["batch_size"]} lr = {config["learning_rate"]}'
    log_dir = os.path.join(config['log_dir'], config['experiment_name'], str(dt.datetime.now()))
    tensorboard = SummaryWriter(comment=comment, log_dir=log_dir)

    copyfile(args.config_path, os.path.join(tensorboard.log_dir, os.path.basename(args.config_path)))

    if config['start_checkpoint']:
        dataloaders = torch.load(os.path.join(config['start_checkpoint'], 'dataloaders.pt'))
    else:
        dataloaders = create_dataloaders()
        torch.save(dataloaders, os.path.join(tensorboard.log_dir, 'dataloaders.pt'))

    epoch_phases = ['train']
    if len(dataloaders['val']) > 0:
        epoch_phases.append('val')

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Nutrition5kModel(config['task_list']).float().to(device)
    # Start training from a checkpoint
    if config['start_checkpoint']:
        previous_epochs = glob(os.path.join(config['start_checkpoint'], 'epochs/*'))
        # Sort epochs
        previous_epochs = sorted(previous_epochs, key=lambda x: int(x))
        best_epoch = previous_epochs[-1]
        model.load_state_dict(torch.load(os.path.join(best_epoch, 'model.pt')))
        optimizer = torch.load(os.path.join(best_epoch, 'optimizer.pt'))
        if config['mixed_precision_enabled']:
            scaler = torch.load(os.path.join(best_epoch, 'scaler.pt'))
        else:
            scaler = None
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'])
        if config['mixed_precision_enabled']:
            scaler = GradScaler()
        else:
            scaler = None

    criterion = n5kloss

    lr_scheduler = ReduceLROnPlateau(optimizer, patience=config['lr_scheduler']['patience'])
    metrics = Metrics(config['task_list'], device, config['prediction_threshold'])

    best_model_path = None

    since = time.time()
    best_val_loss = np.inf
    best_training_loss = np.inf
    for epoch in tqdm(range(config['epochs'])):
        training_loss = None
        val_loss = None
        optimizer.zero_grad(set_to_none=True)
        for phase in epoch_phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            results = run_epoch(model, criterion, dataloaders[phase], device, phase,
                                config['mixed_precision_enabled'], optimizer=optimizer, scaler=scaler,
                                lr_scheduler=lr_scheduler,
                                gradient_acc_steps=config['gradient_acc_steps'],
                                lr_scheduler_metric=config['lr_scheduler']['metric'],
                                task_list=config['task_list'],
                                metrics=metrics)
            if phase == 'train':
                training_loss = results['average loss']
                '''
                for name, weight in model.named_parameters():
                    tensorboard.add_histogram(name, weight, epoch)
                    tensorboard.add_histogram(f'{name}.grad', weight.grad, epoch)
                '''
            else:
                val_loss = results['average loss']

            metrics_results = metrics.compute()
            metrics.reset()

            tensorboard.add_scalar('{} loss'.format(phase), results['average loss'], epoch)
            print('Epoch {} {} loss: {:.4f}'.format(epoch, phase, results['average loss']))

            for task, metric_value in metrics_results.items():
                metric_name = '{} {}'.format(phase, task)
                print('Epoch {} {}: {:.4f}'.format(epoch, metric_name, metric_value))
                tensorboard.add_scalar(metric_name, metric_value, epoch)

        if val_loss and (val_loss < best_val_loss) or (not config['save_best_model_only']):
            epoch_dir = os.path.join(tensorboard.log_dir, 'epoch_{}'.format(epoch))
            os.makedirs(epoch_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(epoch_dir, 'model.pt'))
            torch.save(optimizer, os.path.join(epoch_dir, 'optimizer.pt'))
            if scaler:
                torch.save(scaler, os.path.join(epoch_dir, 'scaler.pt'))

            best_val_loss = val_loss
        if training_loss < best_training_loss:
            best_training_loss = training_loss

        time_elapsed = time.time() - since
        if config['max_training_time'] and (time_elapsed // SECONDS_TO_HOURS) > config['max_training_time']:
            print('Time limit exceeded. Stopping training. ')
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    tensorboard.add_hparams(
        {'learning rate': config['learning_rate'], 'batch size': config['batch_size']},
        {
            'best training loss': best_training_loss,
            'best validation loss': best_val_loss
        },
    )
    tensorboard.close()
