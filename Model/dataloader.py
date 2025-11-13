from pathlib import Path
from typing import Tuple, Optional

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def _parse_label_file(label_path: Path) -> torch.Tensor:
    """Parse a label text file. Returns a 1D float tensor containing all numbers found.

    For YOLO-style multi-line files this will flatten all numeric tokens. Caller
    may interpret them as (class, x, y, w, h) sequences per line.
    """
    if not label_path.exists():
        return torch.tensor([], dtype=torch.float32)
    try:
        txt = label_path.read_text().strip()
    except Exception:
        return torch.tensor([], dtype=torch.float32)
    if txt == "":
        return torch.tensor([], dtype=torch.float32)
    parts = []
    for line in txt.splitlines():
        parts.extend(line.split())
    try:
        vals = [float(x) for x in parts]
    except Exception:
        vals = []
    return torch.tensor(vals, dtype=torch.float32)


class CustomImageDataset(Dataset):
    """Simple image dataset that pairs images with label text files.

    - image_dir: folder with image files
    - label_dir: folder with .txt files (same base filename as image)
    The dataset returns (image_tensor, label_tensor) where label_tensor is a 1D
    float tensor (possibly empty). Images are converted to RGB and transformed
    by the provided torchvision transform.
    """

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        items = sorted(os.listdir(self.image_dir))
        self.image_names = [
            fn
            for fn in items
            if os.path.isfile(self.image_dir / fn) and fn.lower().endswith(valid_exts)
        ]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_names[idx]
        img_path = self.image_dir / img_name
        label_path = self.label_dir / (Path(img_name).stem + ".txt")

        # Open image; fall back to a black image on failure
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # create a black image with a default size (caller should set transform to resize)
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform:
            img = self.transform(img)

        label = _parse_label_file(label_path)
        return img, label # type: ignore


def get_dataloaders(
    data_root: Path,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Build train and validation DataLoaders.

    Returns (train_loader, val_loader, len(train_dataset), len(val_dataset)).
    """
    data_root = Path(data_root)
    image_dir = data_root / images_subdir
    label_dir = data_root / labels_subdir

    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = CustomImageDataset(image_dir, label_dir, transform=transform)
    n = len(dataset)
    if n == 0:
        raise RuntimeError(f"No images found in {image_dir}")

    val_count = int(n * val_split)
    train_count = n - val_count

    # deterministic split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        dataset, [train_count, val_count], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


if __name__ == "__main__":
    # Quick local test (will not run on import)
    base = Path(__file__).resolve().parent.parent / "Data" / "dataset"
    tl, vl, tn, vn = get_dataloaders(base, batch_size=4)
    print(f"Train samples: {tn}, Val samples: {vn}")
    for imgs, labels in tl:
        print("Batch imgs:", imgs.shape, "labels sample:", labels[0])
        break
