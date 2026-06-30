# Mahjong AI

PyTorch utilities for building supervised-learning datasets and models from
Japanese mahjong game records.

The project currently focuses on converting game logs into tensor features for
action prediction. It includes dataset loaders, a rich state encoder, lightweight
mahjong rule helpers, and experimental neural-network models.

## Repository Layout

```text
dataset/
  dataset.py              Dataset extraction, train/validation split helpers
  mahjong_ai_features.py  Game-state encoder: (380, 4, 9) tensors
  mahjong_rules.py        Local rule checks for chi, pon, kan, and red fives
  multizip_dataset.py     Dataset wrapper for multiple ZIP archives
models/
  ResNet.py               ResNet-style classifier
  ViT.py                  Experimental Vision Transformer implementation
train_val.py              Training/evaluation helper draft
test_dataset.py           Small dataset smoke-test script
```

## Data Format

`MahjongDataset` expects a ZIP archive containing `.txt` files. Each text file
must contain JSON with a top-level `log` field. Each item in `log` is a kyoku
round log using kobalab-style record keys such as:

- `qipai`
- `zimo`
- `dapai`
- `fulou`
- `gang`
- `hule`

The dataset returns samples as:

```python
state_tensor, label, action_type
```

where `state_tensor` has shape `(380, 4, 9)`.

Supported action labels:

| Action | Labels |
| --- | --- |
| `dapai` | `0..33`, the discarded tile index with red fives normalized |
| `riichi` | `0` or `1` |
| `fulou` | `0=pass`, `1=chi`, `2=pon`, `3=daiminkan` |
| `gang` | `0=pass/no-gang`, `1=ankan`, `2=kakan` |
| `hule` | `1=win` |

By default, `collect_all_actions=False`, so only `dapai` samples are emitted.

## Setup

Create a Python environment and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy tqdm einops
```

`requirements.txt` is currently empty, so install the packages above manually or
pin them there for your environment.

## Basic Usage

Load discard-prediction samples from one archive:

```python
from dataset.dataset import MahjongDataset, create_dataloaders

dataset = MahjongDataset(
    zip_path="/path/to/data.zip",
    max_files=100,
    collect_all_actions=False,
)

print(dataset.get_statistics())

train_loader, val_loader = create_dataloaders(
    dataset,
    train_ratio=0.9,
    batch_size=64,
    split_by_game=True,
)

state, label, action_type = dataset[0]
print(state.shape)  # torch.Size([380, 4, 9])
print(label, action_type)
```

Load multiple ZIP archives:

```python
from dataset.multizip_dataset import MultiZipMahjongDataset

dataset = MultiZipMahjongDataset(
    ["/path/to/part1.zip", "/path/to/part2.zip"],
    max_files_per_zip=1000,
    collect_all_actions=True,
    include_fulou_negatives=True,
)

print(dataset.get_statistics())
```

Use the ResNet classifier for discard prediction:

```python
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset import MahjongDataset, create_dataloaders
from models.ResNet import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MahjongDataset("/path/to/data.zip", max_files=100)
train_loader, val_loader = create_dataloaders(dataset, batch_size=64)

model = ResNet18(in_channels=380, num_classes=34).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for states, labels, _actions in train_loader:
    states = states.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(states)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

## Notes

- `train_val.py` is a draft and currently references `DATASET_PATH` without
  defining it. Use the examples above as the reliable entry point.
- `models/ViT.py` is experimental and has not been validated as a working
  training model.
- `filter_by_action("discard")` is still accepted as a compatibility alias for
  `filter_by_action("dapai")`.
- `hule` currently emits positive samples only. Negative win-decision samples
  are not synthesized.

## Smoke Test

After placing a compatible data archive somewhere on disk, update the path in
`test_dataset.py` or run an equivalent snippet:

```bash
python test_dataset.py
```

Expected output includes the number of samples, one encoded tensor shape, one
label, and one action type.
