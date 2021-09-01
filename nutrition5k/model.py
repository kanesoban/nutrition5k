import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.inception import InceptionOutputs, InceptionAux


class Nutrition5kModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.inception_v3(pretrained=True)
        self.tasks = ['cal_per_gram_out', 'mass_out']
        # Handle the primary net
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 4096)

        self.task_layers = {}
        for task in self.tasks:
            self.task_layers[task] = [nn.Linear(4096, 4096), nn.Linear(4096, 1)]

        # Handle the auxiliary net for each task
        self.aux_task_logits = {}
        for task in self.tasks:
            self.aux_task_logits[task] = InceptionAux(768, 1)

        '''
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        '''

    def _forward_inception(self, x):
        # N x 3 x 299 x 299
        x = self.base_model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.base_model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.base_model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.base_model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.base_model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.base_model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.base_model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.base_model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.base_model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.base_model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.base_model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = x
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.base_model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.base_model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.base_model.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.base_model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x, aux

    def _forward(self, x):
        x, aux = self._forward_inception(x)
        #x = self.average_pooling(x)
        x = self.fc1(x)
        x = self.fc2(x)

        # Calculate main outputs
        outputs = []
        for task in self.tasks:
            output = x
            for layer in self.task_layers[task]:
                output = layer(output)
            outputs.append(output)

        aux_defined = self.base_model.training and self.base_model.aux_logits

        # Calculate aux outputs
        aux_outputs = []
        if aux_defined:
            for task in self.tasks:
                aux_outputs.append(self.aux_task_logits[task](aux))
        else:
            for task in self.tasks:
                aux_outputs.append(None)

        return outputs, aux

    def forward(self, x):
        x = self.base_model._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.base_model.training and self.base_model.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.base_model.eager_outputs(x, aux)


def create_model(device, train=True):
    model = Nutrition5kModel()
    model = model.to(device)
    if train:
        model.train()
        params_to_update = model.parameters()
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        criterion = nn.MSELoss()
    else:
        model.eval()

    return model
