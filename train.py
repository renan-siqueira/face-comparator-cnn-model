import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import FaceCNN
from data.face_dataset import FaceDataset
from utils.transforms import get_transforms


dataset_path = '/path/to/your/dataset'
batch_size = 4
learning_rate = 0.001
num_epochs = 10

transform = get_transforms()
dataset = FaceDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = FaceCNN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, outputs)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
