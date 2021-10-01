import numpy as np
import torch

torch.set_printoptions(linewidth=120)


def train_step(model, optimizer, criterion, inputs, targets, single_input,
               mixed_precision_enabled, scaler=None, batch_idx=None, gradient_acc_steps=1, metrics=None):
    # Calculate predictions
    with torch.cuda.amp.autocast(enabled=mixed_precision_enabled):
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

    # Backpropagate losses
    if mixed_precision_enabled:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Apply updates
    if (batch_idx + 1) % gradient_acc_steps == 0:
        if mixed_precision_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    if metrics is not None:
        metrics.update(outputs, targets)
    return loss


def eval_step(model, criterion, inputs, targets, single_input, metrics=None):
    # Calculate predictions
    outputs = model(inputs.float())
    if single_input:
        outputs = [outputs[0][:1], outputs[1][:1]]
    outputs = torch.cat(outputs, axis=1)
    # Calculate loss
    loss = criterion(outputs, targets)
    if metrics is not None:
        metrics.update(outputs, targets)
    return loss


def run_epoch(model, criterion, dataloader, device, phase, mixed_precision_enabled,
              optimizer=None, scaler=None, lr_scheduler=None, gradient_acc_steps=None, lr_scheduler_metric='val_loss',
              task_list=('calorie', 'mass', 'fat', 'carb', 'protein'), metrics=None):
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['image']
        target_list = []
        for task in task_list:
            target_list.append(batch[task])

        single_input = inputs.shape[0] == 1
        # Training will not work with bs == 1, so we do a 'hack'
        if single_input:
            dummy_tensor = torch.zeros(batch['image'][:1].shape)
            inputs = torch.cat([batch['image'][:1], dummy_tensor], axis=0)

        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Calculate actual targets
            targets = torch.squeeze(torch.cat(target_list, axis=1))
            if len(targets.shape) == 1:
                targets = torch.unsqueeze(targets, 0)

            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            if phase == 'train':
                loss = train_step(model, optimizer, criterion, inputs,
                                                            targets, single_input,
                                                            mixed_precision_enabled, scaler=scaler,
                                                            batch_idx=batch_idx,
                                                            gradient_acc_steps=gradient_acc_steps, metrics=metrics)
            else:
                loss = eval_step(model, criterion, inputs, targets,
                                                           single_input, metrics=metrics)

        # statistics
        current_loss = loss.item() * inputs.size(0)
        running_loss += current_loss
    if (lr_scheduler_metric == 'val_loss' and phase == 'val') or (
            lr_scheduler_metric == 'train_loss' and phase == 'train'):
        lr_scheduler.step(running_loss)

    results = {
        'average loss': running_loss / len(dataloader.dataset)
    }

    return results
