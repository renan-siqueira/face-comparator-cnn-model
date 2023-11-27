import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.cnn_model import FaceCNN
from src.data.face_dataset import FaceDataset
from src.utils.transforms import get_transforms


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
        loss = criterion(outputs, outputs)  # Ajuste conforme necessário
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:  # Ajuste a frequência de impressão conforme necessário
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

def train_model(model, dataloader, criterion, optimizer, num_epochs, checkpoint_freq, checkpoint_path=None):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        load_checkpoint(checkpoint, model, optimizer)

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch, dataloader, model, criterion, optimizer)
        save_model(model, epoch, f'model_epoch_{epoch}.pth.tar')

        if (epoch) % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pth.tar')


def main():
    dataset_path = 'C:/Users/Meu Computador/Desktop/projetos/00-Datasets/datasets/alissa_white_gluz/002'
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 100
    checkpoint_freq = 50
    checkpoint_path = 'checkpoint.pth.tar'

    transform = get_transforms()
    dataset = FaceDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FaceCNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    train_model(model, dataloader, criterion, optimizer, num_epochs, checkpoint_freq, checkpoint_path)


if __name__ == "__main__":
    main()
