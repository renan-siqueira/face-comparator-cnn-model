import torch


def calculate_similarity(feature1, feature2):
    return torch.cosine_similarity(feature1, feature2, dim=0).item()
