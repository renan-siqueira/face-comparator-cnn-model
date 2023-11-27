import torch
from models.cnn_model import FaceCNN
from utils.transforms import get_transforms
from PIL import Image
from utils.similarity import calculate_similarity


def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def evaluate_similarity(model, image1, image2, transform):
    image1 = load_image(image1, transform)
    image2 = load_image(image2, transform)

    with torch.no_grad():
        feature1 = model(image1)
        feature2 = model(image2)

    similarity = calculate_similarity(feature1, feature2)
    return similarity


transform = get_transforms()
model = FaceCNN()
model.load_state_dict(torch.load('/path/to/your/model.pth'))
model.eval()

similarity = evaluate_similarity(model, '/path/to/image1.jpg', '/path/to/image2.jpg', transform)
print(f'Similarity: {similarity}')
