import numpy as np
import torch

torch.set_printoptions(linewidth=120)


def calculate_correct_predictions(outputs, targets, prediction_threshold):
    outputs_numpy = outputs.cpu().detach().numpy()
    targets_numpy = targets.cpu().detach().numpy()
    return (np.absolute(outputs_numpy - targets_numpy) / targets_numpy) < prediction_threshold


def train_step(model, optimizer, criterion, inputs, targets, single_input, prediction_threshold,
               mixed_precision_enabled, scaler=None, batch_idx=None, gradient_acc_steps=1):
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
    if (batch_idx+1) % gradient_acc_steps == 0:
        if mixed_precision_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    # Calculate 'correct' predictions
    correct_predictions = calculate_correct_predictions(outputs, targets, prediction_threshold)
    return correct_predictions, loss


def eval_step(model, criterion, inputs, targets, single_input, prediction_threshold):
    # Calculate predictions
    outputs = model(inputs.float())
    if single_input:
        outputs = [outputs[0][:1], outputs[1][:1]]
    outputs = torch.cat(outputs, axis=1)
    # Calculate loss
    loss = criterion(outputs, targets)

    # Calculate 'correct' predictions
    # TODO check if you should only calculate this for validation/test
    correct_predictions = calculate_correct_predictions(outputs, targets, prediction_threshold)
    return correct_predictions, loss


def run_epoch(model, criterion, dataloader, device, phase, prediction_threshold, mixed_precision_enabled,
              optimizer=None, scaler=None, lr_scheduler=None, gradient_acc_steps=None, lr_scheduler_metric='val_loss',
              task_list=('calorie', 'mass', 'fat', 'carb', 'protein')):
    running_loss = 0.0
    # Iterate over data.
    correct_predictions = {}
    for task in task_list:
        correct_predictions[task] = 0

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
                task_correct_predictions, loss = train_step(model, optimizer, criterion, inputs, targets, single_input,
                                                       prediction_threshold, mixed_precision_enabled, scaler=scaler,
                                                       batch_idx=batch_idx, gradient_acc_steps=gradient_acc_steps)
            else:
                task_correct_predictions, loss = eval_step(model, criterion, inputs, targets, single_input,
                                                      prediction_threshold)

            for i, task in enumerate(task_list):
                correct_predictions[task] += np.sum(task_correct_predictions[:, i])

        # statistics
        current_loss = loss.item() * inputs.size(0)
        running_loss += current_loss
    if (lr_scheduler_metric == 'val_loss' and phase == 'val') or (lr_scheduler_metric == 'train_loss' and phase == 'train'):
        lr_scheduler.step(running_loss)

    results = {
        'average loss': running_loss / len(dataloader.dataset)
    }

    for task in task_list:
        results['{} prediction accuracy'.format(task)] = correct_predictions[task] / len(dataloader.dataset)

    return results
