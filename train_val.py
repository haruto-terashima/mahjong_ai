import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset import MahjongDataset, create_dataloaders
from models.ResNet import ResNet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MahjongDataset(
    zip_path=DATASET_PATH,
    max_files=50,
    collect_all_actions=False
)

train_loader, val_loader = create_dataloaders(
    dataset, 
    train_ratio=0.9,
    batch_size=64,
    split_by_game=False
    )


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for data, target, _ in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # loss集計
            total_loss += loss.item()

            # accuracy計算
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        acc = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

def evaluate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target, _ in val_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(val_loader)
    acc = correct / total

    return avg_loss, acc

    