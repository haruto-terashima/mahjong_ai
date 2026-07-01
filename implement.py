import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from dataset.dataset import MahjongDataset, create_dataloaders
from models.ResNet import SmallResNet
from models.ViT import ViT
from train_val import evaluate, train_one_epoch


DATA_PATH = ""
# set to model_name to "small_resnet" or "ViT"
model_name = ""
epochs = 10
batch_size = 64
lr = 1e-3
max_files = 50
checkpoint_path_resnet = "small_resnet_dapai.pth"
checkpoint_path_vit = "vit_dapai.pth"
split_by_game = True


def main():
    if not DATA_PATH:
        raise ValueError("Set DATA_PATH in implement.py or pass --data-path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MahjongDataset(
        zip_path=DATA_PATH,
        max_files=max_files,
        collect_all_actions=False,
    )

    train_loader, val_loader = create_dataloaders(
        dataset,
        train_ratio=0.9,
        batch_size=batch_size,
        split_by_game=split_by_game,
    )


    if model_name == "small_resnet":
        model = SmallResNet(
            in_channels=380, 
            num_classes=34
            ).to(device)

    elif model_name == "ViT":
        model = ViT(
            patch_size=1,
            n_classes=34,
            dim=64,
            depth=6,
            n_heads=8,
        ).to(device)
    else:
        raise ValueError('model_name must be "small_resnet" or "ViT"')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    save_path = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if model_name == "small_resnet":
                save_path = Path(checkpoint_path_resnet)
            elif model_name == "ViT":
                save_path = Path(checkpoint_path_vit)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                save_path,
            )

    print(f"Best Val Acc: {best_val_acc:.4f}")
    if save_path is not None:
        print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
