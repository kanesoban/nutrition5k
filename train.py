import argparse
import time
import os

TEST_MODE = True
import numpy as np
import torch

if TEST_MODE:
    np.random.seed(0)
    torch.manual_seed(0)
from torch.utils.data import DataLoader
from torchvision import transforms

torch.set_printoptions(linewidth=120)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from nutrition5k.dataset import Rescale, ToTensor, Nutrition5kDataset
from nutrition5k.model import Nutrition5kModel


def parse_args():
    """ Parse the arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', help='Name of the base config file without extension.', required=True)
    return parser.parse_args()


def train_step(model, inputs, targets, single_input):
    # Calculate predictions
    outputs, aux_outputs = model(inputs.float())
    if single_input:
        outputs = [outputs[0][:1], outputs[1][:1]]
    outputs = torch.cat(outputs, axis=1)
    if single_input:
        aux_outputs = [aux_outputs[0][:1], aux_outputs[1][:1]]
    aux_outputs = torch.cat(aux_outputs, axis=1)
    # Calculate losses
    loss_main = criterion(outputs, targets)
    loss_aux = criterion(aux_outputs, targets)
    loss = loss_main + 0.4 * loss_aux
    loss.backward()
    optimizer.step()
    return loss


def eval_step(model, inputs, targets, single_input):
    # Calculate predictions
    outputs = model(inputs.float())
    if single_input:
        outputs = [outputs[0][:1], outputs[1][:1]]
    outputs = torch.cat(outputs, axis=1)
    # Calculate loss
    loss = criterion(outputs, targets)
    return loss


def run_epoch(model, dataloader):
    running_loss = 0.0
    # Iterate over data.
    for batch in dataloader:
        inputs = batch['image'].to(device)
        mass = batch['mass'].to(device)
        calories = batch['calories'].to(device)

        single_input = inputs.shape[0] == 1
        # Training will not work with bs == 1, so we do a 'hack'
        if single_input:
            dummy_tensor = torch.zeros(batch['image'][:1].shape)
            inputs = torch.cat([batch['image'][:1], dummy_tensor], axis=0)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Calculate actual targets
            targets = torch.squeeze(torch.cat([mass, calories], axis=1))
            if len(targets.shape) == 1:
                targets = torch.unsqueeze(targets, 0)

            if phase == 'train':
                loss = train_step(model, inputs, targets, single_input)
            else:
                loss = eval_step(model, inputs, targets, single_input)

        # statistics
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloaders[phase].dataset)


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    with open(args.config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    transformed_dataset = Nutrition5kDataset(config['dataset_dir'], transform=transforms.Compose(
        [Rescale((299, 299)), ToTensor()]))

    n_videos = len(transformed_dataset)
    train_size = int(config['split']['train'] * n_videos)
    val_size = int(config['split']['validation'] * n_videos)
    test_size = n_videos - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(transformed_dataset, [train_size, val_size, test_size])
    epoch_phases = ['train', 'val']
    dataloaders = {
        'train': DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,
                            num_workers=config['dataset_workers']),
        'val': DataLoader(val_set, batch_size=config['batch_size'], shuffle=False,
                          num_workers=config['dataset_workers']),
        'test': DataLoader(val_set, batch_size=config['batch_size'], shuffle=False,
                           num_workers=config['dataset_workers'])
    }

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Nutrition5kModel().to(device)
    # Start training from a checkpoint
    if config['start_checkpoint']:
        model.load_state_dict(torch.load(config['start_checkpoint']))

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.L1Loss()
    comment = f' batch_size = {config["batch_size"]} lr = {config["learning_rate"]}'
    tensorboard = SummaryWriter(comment=comment)
    best_model_path = None

    since = time.time()
    best_val_loss = np.inf
    best_training_loss = np.inf
    for epoch in tqdm(range(config['epochs'])):
        training_loss = None
        val_loss = None
        for phase in epoch_phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            epoch_loss = run_epoch(model, dataloaders[phase])
            if phase == 'train':
                training_loss = epoch_loss
                '''
                for name, weight in model.named_parameters():
                    tensorboard.add_histogram(name, weight, epoch)
                    tensorboard.add_histogram(f'{name}.grad', weight.grad, epoch)
                '''
            else:
                val_loss = epoch_loss

            print('{} loss: {:.4f}'.format(phase, epoch_loss))

        tensorboard.add_scalar("Training loss", training_loss, epoch)
        tensorboard.add_scalar("Validation loss", val_loss, epoch)

        if (val_loss < best_val_loss) or (not config['save_best_model_only']):
            torch.save(model.state_dict(), os.path.join(config['model_save_path'], 'epoch_{}.pt'.format(epoch)))
            best_val_loss = val_loss
        if (training_loss < best_val_loss) or (not config['save_best_model_only']):
            best_training_loss = training_loss
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
