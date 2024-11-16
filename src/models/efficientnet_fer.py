import os

import numpy as np
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
from scipy.special import softmax

from src.utils import setup_logger

logger = setup_logger()

class EfficientNetFER(nn.Module):
    def __init__(self, model_path="efficientnet_b3", num_classes=7, verbose=True):
        """
        Initializes EfficientNetFER for both training and inference.

        - If the model path ends with ".onnx", it loads an ONNX model for inference.
        - Otherwise, it loads a PyTorch model (with or without pretrained weights).

        :param model_path: Path to a PyTorch model file, ONNX model file, or model name for training.
        :param num_classes: Number of output classes (default: 7).
        :param verbose: If True, enables detailed logging; if False, only essential logs are printed.
        """
        super(EfficientNetFER, self).__init__()
        self.verbose = verbose

        self.onnx_mode = model_path.endswith(".onnx")
        self.model_path = model_path

        if self.onnx_mode:
            if self.verbose:
                logger.info(f"Loading ONNX model from {model_path}...")

            if not os.path.exists(model_path):
                logger.error(f"ONNX model file '{model_path}' not found!")
                raise FileNotFoundError(f"ONNX model file '{model_path}' not found!")

            # Check available providers and set execution backend
            available_providers = ort.get_available_providers()
            use_gpu = ("CUDAExecutionProvider" in available_providers) and torch.cuda.is_available()
            providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

            try:
                self.model = ort.InferenceSession(model_path, providers=providers)
                if self.verbose:
                    logger.info(f"ONNX model loaded successfully from {model_path} using {'GPU' if use_gpu else 'CPU'}")
            except Exception as e:
                logger.error(f"Error loading ONNX model: {e}", exc_info=True)
                raise


        else:
            if self.verbose:
                logger.info(f"Loading PyTorch model: {model_path}")

            try:
                self.backbone = timm.create_model(model_path, pretrained=True)
                in_features = self.backbone.classifier.in_features

                # Replace classifier for emotion classification
                self.backbone.classifier = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )

                if os.path.isfile(model_path):
                    try:
                        self.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

                        if self.verbose:
                            logger.info(f"PyTorch model weights loaded successfully from {model_path}")

                    except Exception as e:
                        logger.error(f"Error loading PyTorch model weights: {e}", exc_info=True)
                        raise

                self.eval()
                if self.verbose:
                    logger.info("Model set to evaluation mode by default.")

            except Exception as e:
                logger.error(f"Error loading PyTorch model: {e}", exc_info=True)
                raise

        if self.verbose:
            logger.info("EfficientNetFER model initialized successfully.")

    def forward(self, x):
        if self.onnx_mode:
            logger.error("Forward pass not available in ONNX mode. Use __call__() instead.")
            raise RuntimeError("This model is in ONNX mode. Use __call__() instead.")

        return self.backbone(x)

    def __call__(self, x, is_inference=True):
        if self.onnx_mode:
            return self._run_onnx_inference(x)
        else:
            if self.training:
                return self._run_pytorch_train(x)  # Use training mode function
            return self._run_pytorch_inference(x, is_inference)  # Default inference

    def _run_pytorch_train(self, x):
        """
        Runs forward pass when the model is in training mode.

        :param x: Input image tensor.
        :return: Model logits (before softmax).
        """
        return self.forward(x)

    def _run_pytorch_inference(self, x, is_inference):
        """
        Runs inference using the PyTorch model.

        :param x: Input image tensor (or NumPy array).
        :return: Model predictions.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if not isinstance(x, torch.Tensor):
            logger.error("Input must be a PyTorch Tensor!")
            raise TypeError("Input must be a PyTorch Tensor!")

        device = next(self.backbone.parameters()).device
        x = x.to(device)

        if self.training:
            logger.warning("Model is in training mode during inference. Switching to evaluation mode.")
            self.eval()

        with torch.no_grad():
            if self.verbose:
                logger.debug("Running PyTorch model inference...")

            outputs = self.forward(x)
            if not is_inference:
                return outputs
            outputs = softmax(outputs, axis=1)
            class_ids = np.argmax(outputs, axis=1)
            return class_ids

    def _run_onnx_inference(self, x):
        """
        Runs inference using the ONNX model.

        :param x: Input image batch as a NumPy array.
        :return: Predicted class IDs.
        """
        if not isinstance(x, np.ndarray):
            logger.error("Input must be a NumPy array for ONNX models!")
            raise TypeError("Input must be a NumPy array for ONNX models!")

        input_shape = self.model.get_inputs()[0].shape  # Expected shape from ONNX model
        batch_size, c, h, w = input_shape

        if x.ndim == 3 and x.shape == (c, h, w):
            if self.verbose:
                logger.warning("Received a single image (3D). Expanding batch dimension.")
            x = np.expand_dims(x, axis=0)  # Convert (3, H, W) to (1, 3, H, W)

        elif x.ndim != 4 or x.shape[1:] != (c, h, w):
            logger.error(f"Invalid input shape for ONNX model. Expected: {input_shape}, Got: {x.shape}")
            raise ValueError(f"Expected input shape (batch_size, {c}, {h}, {w}), but got {x.shape}")

        try:
            inputs = {self.model.get_inputs()[0].name: x}
            outputs = self.model.run(None, inputs)[0]  # Get ONNX model output

            if self.verbose:
                logger.debug(f"ONNX raw output shape: {outputs.shape}")

            # Apply softmax and get class predictions
            outputs = softmax(outputs, axis=1)
            class_ids = np.argmax(outputs, axis=1)

            return class_ids  # Return batch of predicted classes

        except Exception as e:
            logger.error(f"Error during ONNX model inference: {e}", exc_info=True)
            raise