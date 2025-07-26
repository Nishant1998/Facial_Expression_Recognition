import sys
from pathlib import Path

import argparse
import torch

from src.data.dataset.loader import get_data_loader
from src.models.efficientnet_fer import EfficientNetFER
from src.train.evaluation import evaluate_model_performance
from src.utils.config_utils import load_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))



def export_to_onnx(model_name="efficientnet_b3", num_classes=7,
                   model_path="src/models/weights/efficientnet_b0_fer.pth",
                   output_path="src/models/weights/EfficientNetFER.onnx"):
    model = EfficientNetFER(model_path=model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.backbone
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export EfficientNetFER to ONNX format")
    parser.add_argument("--config", type=str, default="src/config/custom.yaml", help="Path to the configuration YAML file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, classes = get_data_loader(cfg)

    if cfg.EXPORT.EVAL:
        model = EfficientNetFER(model_path=cfg.EXPORT.MODEL_NAME, num_classes=classes).to(device)
        model.load_state_dict(torch.load(cfg.EXPORT.INPUT_WEIGHTS, map_location=torch.device("cpu")))
        evaluate_model_performance(model, test_loader, device)


    export_to_onnx(model_name=cfg.TRAIN.MODEL_NAME,
                   num_classes=classes,
                   model_path=cfg.EXPORT.INPUT_WEIGHTS,
                   output_path=cfg.EXPORT.OUTPUT_PATH)

    if cfg.EXPORT.EVAL:
        model = EfficientNetFER(model_path=cfg.EXPORT.OUTPUT_PATH, num_classes=classes).to(device)
        evaluate_model_performance(model, test_loader, device)