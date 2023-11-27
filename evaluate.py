import torch
from models.cnn_model import FaceCNN
from utils.transforms import get_transforms
from PIL import Image
from utils.similarity import calculate_similarity

def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Adiciona uma dimensão de batch
    return image

def evaluate_similarity(model, image_path1, image_path2, transform):
    image1 = load_image(image_path1, transform)
    image2 = load_image(image_path2, transform)

    with torch.no_grad():
        feature1 = model(image1)
        feature2 = model(image2)

    similarity = calculate_similarity(feature1, feature2)
    return similarity

def main():
    checkpoint_path = '/path/to/your/model_checkpoint.pth'
    image_path1 = '/path/to/image1.jpg'
    image_path2 = '/path/to/image2.jpg'

    model = FaceCNN()
    model = load_model(checkpoint_path, model)
    model.eval()

    transform = get_transforms()
    similarity = evaluate_similarity(model, image_path1, image_path2, transform)
    print(f'Similarity: {similarity}')

if __name__ == "__main__":
    main()
