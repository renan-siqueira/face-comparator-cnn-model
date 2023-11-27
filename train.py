import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import FaceCNN
from data.face_dataset import FaceDataset
from utils.transforms import get_transforms

# Configurações iniciais
dataset_path = '/path/to/your/dataset'
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# Preparar dataset e dataloader
transform = get_transforms()
dataset = FaceDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Inicializar modelo e otimizador
model = FaceCNN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()  # Exemplo de função de perda

# Loop de treinamento
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, outputs)  # Substitua por uma função de perda adequada
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
