import torch
import torch.onnx

from src.data.dataset.loader import get_data_loader
from src.models.efficientnet_fer import EfficientNetFER


def train_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_data_loader(cfg)
    model = EfficientNetFER(model_path=cfg.TRAIN.MODEL_NAME, num_classes=classes).to(device)


