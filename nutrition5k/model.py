import warnings

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.models.inception import InceptionOutputs, InceptionAux


class InceptionAuxNutrition5k(InceptionAux):
    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = F.relu(self.fc(x))
        # N x 1000
        return x


class Nutrition5kModel(nn.Module):
    def __init__(self, tasks, use_end_relus=True):
        super().__init__()
        self.base_model = torchvision.models.inception_v3(pretrained=True)
        self.tasks = tasks
        self.use_end_relus = use_end_relus
        # Handle the primary net
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 4096)

        self.task_layers = {}
        for task in self.tasks:
            self.task_layers[task] = [nn.Linear(4096, 4096), nn.Linear(4096, 1)]

        # Handle the auxiliary net for each task
        self.aux_task_logits = {}
        for task in self.tasks:
            if self.use_end_relus:
                self.aux_task_logits[task] = InceptionAuxNutrition5k(768, 1)
            else:
                self.aux_task_logits[task] = InceptionAux(768, 1)

    def float(self):
        super().float()
        for task in self.tasks:
            for layer in self.task_layers[task]:
                for param in layer.parameters():
                    param.float()

        for task in self.tasks:
            for param in self.aux_task_logits[task].parameters():
                param.float()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # We need to manually send some layers to the appropriate device
        for task in self.tasks:
            for layer in self.task_layers[task]:
                layer.to(*args, **kwargs)

        for task in self.tasks:
            self.aux_task_logits[task].to(*args, **kwargs)
        return self

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Calculate main outputs
        outputs = []
        for task in self.tasks:
            output = x
            if self.use_end_relus:
                output = F.relu(self.task_layers[task][0](output))
                output = self.task_layers[task][1](output)
            else:
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
            for _ in self.tasks:
                aux_outputs.append(None)

        return outputs, aux_outputs

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
