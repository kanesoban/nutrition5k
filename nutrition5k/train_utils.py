import numpy as np
import torch

torch.set_printoptions(linewidth=120)


def calculate_correct_predictions(outputs, targets, prediction_threshold):
    outputs_numpy = outputs.detach().numpy()
    targets_numpy = targets.detach().numpy()
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
    correct_predictions = calculate_correct_predictions(outputs, targets, prediction_threshold)
    return correct_predictions, loss


def run_epoch(model, criterion, dataloader, device, phase, prediction_threshold, mixed_precision_enabled,
              optimizer=None, scaler=None, gradient_acc_steps=None):
    running_loss = 0.0
    # Iterate over data.
    mass_correct_predictions = 0
    calories_correct_predictions = 0
    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['image'].to(device)
        mass = batch['mass'].to(device)
        calories = batch['calories'].to(device)

        single_input = inputs.shape[0] == 1
        # Training will not work with bs == 1, so we do a 'hack'
        if single_input:
            dummy_tensor = torch.zeros(batch['image'][:1].shape)
            inputs = torch.cat([batch['image'][:1], dummy_tensor], axis=0)

        # zero the parameter gradients
        if phase == 'train':
            optimizer.zero_grad(set_to_none=True)

        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Calculate actual targets
            targets = torch.squeeze(torch.cat([mass, calories], axis=1))
            if len(targets.shape) == 1:
                targets = torch.unsqueeze(targets, 0)

            if phase == 'train':
                correct_predictions, loss = train_step(model, optimizer, criterion, inputs, targets, single_input,
                                                       prediction_threshold, mixed_precision_enabled, scaler=scaler,
                                                       batch_idx=batch_idx, gradient_acc_steps=gradient_acc_steps)
            else:
                correct_predictions, loss = eval_step(model, criterion, inputs, targets, single_input,
                                                      prediction_threshold)
            mass_correct_predictions += np.sum(correct_predictions[:, 0])
            calories_correct_predictions += np.sum(correct_predictions[:, 1])

        # statistics
        running_loss += loss.item() * inputs.size(0)

    return {
        'average loss': running_loss / len(dataloader.dataset),
        'mass prediction accuracy': mass_correct_predictions / len(dataloader.dataset),
        'calorie prediction accuracy': calories_correct_predictions / len(dataloader.dataset)
    }
