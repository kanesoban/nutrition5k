import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

from nutrition5k.dataset import Rescale, ToTensor, Nutrition5kDataset
from nutrition5k.model import Nutrition5kModel


def parse_args():
    """ Parse the arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', help='Name of the base config file without extension.', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    with open(args.config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    transformed_dataset = Nutrition5kDataset(config['dataset_dir'], transform=transforms.Compose(
        [Rescale((299, 299)), ToTensor()]))

    n_videos = len(transformed_dataset)
    train_size = int(config['split'] * n_videos)
    val_size = n_videos - train_size

    train_set, val_set = torch.utils.data.random_split(transformed_dataset, [train_size, val_size])
    phases = ['train', 'val']
    dataloaders = {
        'train': DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['dataset_workers']),
        'val': DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['dataset_workers'])
    }

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Nutrition5kModel().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.L1Loss()

    since = time.time()
    for epoch in tqdm(range(config['epochs'])):
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            # Iterate over data.
            for batch in dataloaders[phase]:
                inputs = batch['image'].to(device)
                mass = batch['mass'].to(device)
                calories = batch['calories'].to(device)

                single_input = inputs.shape[0] == 1
                # Training will not work with bs == 1, so we do a 'hack'
                if single_input:
                    dummy_tensor = torch.zeros(batch['image'][:1].shape)
                    inputs = torch.cat([batch['image'][:1], dummy_tensor], axis=0)

                running_loss = 0.0

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
                    else:
                        # Calculate predictions
                        outputs = model(inputs.float())
                        if single_input:
                            outputs = [outputs[0][:1], outputs[1][:1]]
                        outputs = torch.cat(outputs, axis=1)

                        # Calculate loss
                        loss = criterion(outputs, targets)
                    preds = outputs

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
