import os
import torch
from PIL import Image

from src.settings import config
from src.models.cnn_model import FaceCNN
from src.utils.transforms import get_transforms
from src.utils.similarity import calculate_similarity


def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
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
    version = 'v1'

    checkpoint_path = os.path.join(
        config.APP_PATH_DATA,
        version,
        config.APP_PATH_CHECKPOINT_FILENAME
    )

    training_person_path = 'person-training.jpg'
    other_person_path = 'other-person.jpg'

    model = FaceCNN()
    model = load_model(checkpoint_path, model)
    model.eval()

    transform = get_transforms()
    similarity = evaluate_similarity(model, training_person_path, other_person_path, transform)
    print(f'Similarity: {similarity}')


if __name__ == "__main__":
    main()
