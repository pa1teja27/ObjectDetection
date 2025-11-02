import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class ObjectDetectionDataset(Dataset):
    """
    PyTorch Dataset for loading YOLO-format object detection data.
    
    Directory structure expected:
        dataset/
            images/
                img1.jpg
                img2.jpg
            labels/
                img1.txt
                img2.txt
    
    Each label file contains: class_id x_center y_center width height (normalized 0-1)
    """
    
    def __init__(self, images_dir, labels_dir, transform=None, img_size=(256, 256)):
        """
        Args:
            images_dir: Path to images folder
            labels_dir: Path to labels folder
            transform: Optional transforms to apply to images
            img_size: Target image size (H, W)
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_size = img_size
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        # Filter to only images that have corresponding labels
        self.valid_samples = []
        for img_file in self.image_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                self.valid_samples.append(img_file)
        
        print(f"Loaded {len(self.valid_samples)} samples from {images_dir}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (C, H, W)
            target: Dict containing:
                'boxes': Tensor of shape (N, 4) - [x_center, y_center, width, height] normalized
                'labels': Tensor of shape (N,) - class labels
        """
        img_file = self.valid_samples[idx]
        img_path = os.path.join(self.images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_file)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load labels
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    boxes.append([x_center, y_center, width, height])
                    labels.append(class_id)
        
        # Convert to tensors
        if len(boxes) == 0:
            # If no boxes, create empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = transforms.ToTensor()(image)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': idx,
            'image_file': img_file
        }
        
        return image, target


def get_data_loaders(data_root='../Data/dataset', batch_size=8, img_size=(256, 256), num_workers=0):
    """
    Create training and validation data loaders.
    
    Args:
        data_root: Root directory containing images/ and labels/ folders
        batch_size: Batch size for training
        img_size: Image size (H, W)
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    images_dir = os.path.join(data_root, 'images')
    labels_dir = os.path.join(data_root, 'labels')
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = ObjectDetectionDataset(images_dir, labels_dir, transform=train_transform, img_size=img_size)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """
    Custom collate function to handle variable number of boxes per image.
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, 0)
    return images, targets


# Simple CNN model for object detection (example)
class SimpleDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simple predictor for one object per image
        # Output: [class_logits (num_classes), bbox (4)]
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes + 4)  # class + bbox
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        features = self.features(x)
        output = self.predictor(features)
        
        # Split into class logits and bbox
        class_logits = output[:, :self.num_classes]
        bbox = output[:, self.num_classes:]
        
        return class_logits, bbox


if __name__ == '__main__':
    # Test the dataset loader
    print("Testing ObjectDetectionDataset...")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        data_root='../Data/dataset',
        batch_size=4,
        img_size=(256, 256),
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading a batch
    images, targets = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Number of targets: {len(targets)}")
    
    for i, target in enumerate(targets[:2]):  # Show first 2
        print(f"\nSample {i}:")
        print(f"  Image file: {target['image_file']}")
        print(f"  Boxes shape: {target['boxes'].shape}")
        print(f"  Labels: {target['labels']}")
        print(f"  Boxes: {target['boxes']}")
    
    # Test model
    print("\n\nTesting SimpleDetector model...")
    model = SimpleDetector(num_classes=2)
    class_logits, bbox = model(images)
    print(f"Class logits shape: {class_logits.shape}")
    print(f"BBox predictions shape: {bbox.shape}")
    
    print("\n✅ Dataset and model test complete!")
