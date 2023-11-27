import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.settings import config
from src.models.cnn_model import FaceCNN
from src.data.face_dataset import FaceDataset
from src.utils.transforms import get_transforms


def load_params(params_path):
    with open(params_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_model(model, epoch, filename="model_epoch.pth.tar"):
    state = {'epoch': epoch, 'state_dict': model.state_dict()}
    torch.save(state, filename)


def save_checkpoint(model, optimizer, epoch, filename="checkpoint_epoch.pth.tar"):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Carregando checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train_one_epoch(epoch, dataloader, model, criterion, optimizer):
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, outputs)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')


def train_model(checkpoints_dir, model, dataloader, criterion, optimizer, num_epochs, checkpoint_freq, checkpoint_path=None):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        load_checkpoint(checkpoint, model, optimizer)

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch, dataloader, model, criterion, optimizer)
        checkpoint_to_save = os.path.join(checkpoints_dir, 'checkpoint.pth.tar')
        save_model(model, epoch, f'model_epoch_{epoch}.pth.tar')

        if (epoch) % checkpoint_freq == 0:
            checkpoint_to_save = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pth.tar')
            save_checkpoint(model, optimizer, epoch, checkpoint_to_save)


def main():
    params = load_params(config.APP_PATH_PARAMS_FILE)

    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs']
    checkpoint_freq = params['checkpoint_freq']
    version = params['version']

    checkpoints_dir = os.path.join(config.APP_PATH_DATA, version, 'weights')
    os.makedirs(checkpoints_dir, exist_ok=True)

    dataset_path = config.APP_PATH_DATASET
    checkpoint_path = os.path.join(checkpoints_dir, 'checkpoint.pth.tar')

    transform = get_transforms()
    dataset = FaceDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FaceCNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    train_model(
        checkpoints_dir,
        model,
        dataloader,
        criterion,
        optimizer,
        num_epochs,
        checkpoint_freq,
        checkpoint_path
    )


if __name__ == "__main__":
    main()
