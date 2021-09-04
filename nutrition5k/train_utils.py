import torch

torch.set_printoptions(linewidth=120)


def train_step(model, optimizer, criterion, inputs, targets, single_input):
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


def eval_step(model, criterion, inputs, targets, single_input):
    # Calculate predictions
    outputs = model(inputs.float())
    if single_input:
        outputs = [outputs[0][:1], outputs[1][:1]]
    outputs = torch.cat(outputs, axis=1)
    # Calculate loss
    loss = criterion(outputs, targets)
    return loss


def run_epoch(model, criterion, dataloader, device, phase, optimizer=None):
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
        if phase == 'train':
            optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Calculate actual targets
            targets = torch.squeeze(torch.cat([mass, calories], axis=1))
            if len(targets.shape) == 1:
                targets = torch.unsqueeze(targets, 0)

            if phase == 'train':
                loss = train_step(model, optimizer, criterion, inputs, targets, single_input)
            else:
                loss = eval_step(model, criterion, inputs, targets, single_input)

        # statistics
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)
