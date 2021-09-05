import time

import torch

torch.set_printoptions(linewidth=120)
import yaml

from nutrition5k.model import Nutrition5kModel
from nutrition5k.train_utils import run_epoch
from nutrition5k.utils import parse_args


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    with open(args.config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    dataloader = torch.load('test_loader.pt')

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Nutrition5kModel().to(device)
    # Load a checkpoint
    model.load_state_dict(torch.load(config['test_checkpoint']))

    criterion = torch.nn.L1Loss()

    since = time.time()
    model.eval()
    results = run_epoch(model, criterion, dataloader, device, 'test', False, config['prediction_threshold'])
    time_elapsed = time.time() - since
    print('{} loss: {:.4f}'.format('Test', results['average loss']))
    print('{} mass prediction accuracy: {:.4f}'.format('Test', results['mass prediction accuracy']))
    print('{} calories prediction accuracy: {:.4f}'.format('Test', results['calories prediction accuracy']))
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
