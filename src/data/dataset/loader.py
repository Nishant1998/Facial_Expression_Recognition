import cv2
import numpy as np
import torch
import torch.onnx
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset.dataset import RAFDBDataset

train_transforms = train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), value=0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_collate_fn(batch):
    """
    Custom collate function to load images inside the dataloader.

    :param batch: List of (image_path, label) tuples.
    :return: Batched images and labels as tensors.
    """
    transform = train_transforms

    image_paths, labels = zip(*batch)
    images = []

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        images.append(image)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def test_collate_fn(batch):
    """
    Custom collate function to load images inside the dataloader.

    :param batch: List of (image_path, label) tuples.
    :return: Batched images and labels as tensors.
    """
    transform = test_transforms

    image_paths, labels = zip(*batch)
    images = []

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        images.append(image)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def get_data_loader(cfg):
    train_dataset = RAFDBDataset(csv_file=f'{cfg.DATASET.PATH}/raf-db-dataset/train_labels.csv',
                                 root_dir=f'{cfg.DATASET.PATH}/raf-db-dataset/DATASET/train')

    test_dataset = RAFDBDataset(csv_file=f'{cfg.DATASET.PATH}/raf-db-dataset/test_labels.csv',
                                root_dir=f'{cfg.DATASET.PATH}/raf-db-dataset/DATASET/test')

    # Compute class weights for balanced sampling
    labels = np.array(train_dataset.labels['label'])
    classes = len(set(labels))
    class_sample_counts = np.bincount(labels)
    weights = 1.0 / class_sample_counts[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True,
                              collate_fn=train_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                             collate_fn=test_collate_fn)

    return train_loader, test_loader, classes