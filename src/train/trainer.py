import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from src.data.dataset.loader import get_data_loader
from src.models.efficientnet_fer import EfficientNetFER
from src.train.evaluation import evaluate_model_performance

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, False)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

def train_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_data_loader(cfg)
    model = EfficientNetFER(model_path=cfg.TRAIN.MODEL_NAME, num_classes=classes).to(device)

    # loss function
    class_counts = train_loader.dataset.labels.iloc[:, 1].value_counts().sort_index().values
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.25)  # Reduce LR by 50% every 10 epochs

    # Training Loop
    num_epochs = cfg.TRAIN.EPOCHS
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        scheduler.step()  # Reduce LR every 10 epochs

        # Print Training Accuracy
        train_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_acc:.2f}%")

        # Run Test Loop Every 5 Epochs
        if (epoch + 1) % 5 == 0:
            test_acc = test_model(model, test_loader)
            print(f"Test Accuracy after Epoch {epoch + 1}: {test_acc:.2f}%")
            output_path = f"src/models/weights/{cfg.TRAIN.MODEL_NAME}_fer_test_{epoch + 1}{test_acc:.2f}.pth"
            torch.save(model.state_dict(), output_path)
            print(f"✅ Model saved successfully, at : {output_path}")

    # Save trained model
    output_path = f"src/models/weights/{cfg.TRAIN.MODEL_NAME}_fer.pth"
    torch.save(model.state_dict(), output_path)
    print(f"✅ Model saved successfully, at : {output_path}")

    evaluate_model_performance(model, test_loader, device)
