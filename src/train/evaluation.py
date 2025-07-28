import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.onnx
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

def evaluate_model_performance(model, test_loader, device):
    print("🔍 Evaluating model performance...")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            labels_np = labels.numpy() if model.onnx_mode else labels.cpu().numpy()
            all_labels.extend(labels_np)

            if model.onnx_mode:
                images_np = images.numpy()
                preds = model(images_np)
                all_preds.extend(preds)
                correct += np.sum(preds == labels_np)
                total += len(labels_np)
            else:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, is_inference=False)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"F1 Score: {f1:.2f}")

    print("\n📌 Precision per class:")
    for i, p in enumerate(precision):
        print(f" - Class {i}: {p:.2f}")

    print("\n📌 Recall per class:")
    for i, r in enumerate(recall):
        print(f" - Class {i}: {r:.2f}")

    class_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Percentage)")
    plt.show()

    return accuracy, precision, recall, f1

