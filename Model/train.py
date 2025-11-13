import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import json
import os

# --- Import from our custom files ---
# Assumes model.py, loss.py, and dataset.py are in the same directory
try:
    from model import YOLOv8FromScratch
    from loss import v8DetectionLoss
    from dataloader import CustomImageDataset
except ImportError:
    print("Error: Make sure model.py, loss.py, and dataset.py are in the same directory.")
    exit(1)


def detection_collate_fn(batch):
    """
    Custom collate function for object detection.
    """
    images, labels = zip(*batch)
    
    batched_images = torch.stack(images, 0)
    batched_targets = []
    
    # Loop over each item in the batch
    for i, label_tensor in enumerate(labels):
        num_objects = label_tensor.numel() // 5
        if num_objects == 0:
            continue
        
        label_data = label_tensor.view(num_objects, 5)
        batch_idx = torch.full((num_objects, 1), float(i))
        label_with_batch_idx = torch.cat([batch_idx, label_data], dim=1)
        
        batched_targets.append(label_with_batch_idx)

    # Concatenate all target tensors from all images in the batch
    if len(batched_targets) > 0:
        batched_targets = torch.cat(batched_targets, 0)
    else:
        batched_targets = torch.empty(0, 6)
        
    return {'img': batched_images, 'targets': batched_targets}


def get_dataloaders_for_detection(data_root, img_size, batch_size, val_split=0.2, seed=42):
    """
    Creates train and validation DataLoaders with the correct collate_fn.
    """
    
    image_dir = Path(data_root) / "images"
    label_dir = Path(data_root) / "labels"
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    dataset = CustomImageDataset(image_dir, label_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
        
    n = len(dataset)
    val_count = int(n * val_split)
    train_count = n - val_count
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, 
        pin_memory=True,
        collate_fn=detection_collate_fn 
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )
    
    return train_loader, val_loader


def run_training(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    """
    The main training loop.
    """
    
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loss_cls_total = 0.0
        loss_iou_total = 0.0
        loss_dfl_total = 0.0
        
        # --- NEW: Accuracy trackers for training ---
        running_correct_cls = 0.0
        running_num_positives = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            batch['img'] = batch['img'].to(device)
            batch['targets'] = batch['targets'].to(device)
            
            pred_bboxes_dist, pred_classes = model(batch['img'])
            
            # --- MODIFIED: Unpack new accuracy tuple ---
            total_loss, (loss_cls, loss_iou, loss_dfl), (num_correct, num_pos) = criterion(
                (pred_bboxes_dist, pred_classes), 
                batch  
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # --- Log losses ---
            running_loss += total_loss.item()
            loss_cls_total += loss_cls
            loss_iou_total += loss_iou
            loss_dfl_total += loss_dfl
            
            # --- NEW: Log accuracy ---
            running_correct_cls += num_correct
            running_num_positives += num_pos
            
            train_pbar.set_postfix(
                loss=f"{total_loss.item():.4f}", 
                avg_loss=f"{running_loss / (train_pbar.n + 1):.4f}"
            )
        
        # --- Calculate epoch averages ---
        avg_train_loss = running_loss / len(train_loader)
        avg_cls_loss = loss_cls_total / len(train_loader)
        avg_iou_loss = loss_iou_total / len(train_loader)
        avg_dfl_loss = loss_dfl_total / len(train_loader)
        # --- NEW: Calculate train accuracy ---
        train_acc = (running_correct_cls / running_num_positives * 100) if running_num_positives > 0 else 0.0
        
        # --- MODIFIED: Updated print statement ---
        print(
            f"Epoch {epoch+1} Train Summary: "
            f"Total Loss: {avg_train_loss:.4f}, "
            f"Cls Acc: {train_acc:.2f}%, "
            f"(Cls: {avg_cls_loss:.4f}, IoU: {avg_iou_loss:.4f}, DFL: {avg_dfl_loss:.4f})"
        )
        
        # --- Validation loop ---
        model.eval()
        val_loss = 0.0
        # --- NEW: Accuracy trackers for validation ---
        val_correct_cls = 0.0
        val_num_positives = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                batch['img'] = batch['img'].to(device)
                batch['targets'] = batch['targets'].to(device)
                
                pred_bboxes_dist, pred_classes = model(batch['img'])
                
                # --- MODIFIED: Unpack accuracy tuple ---
                total_loss, _, (num_correct, num_pos) = criterion(
                    (pred_bboxes_dist, pred_classes), 
                    batch 
                )
                
                val_loss += total_loss.item()
                # --- NEW: Log validation accuracy ---
                val_correct_cls += num_correct
                val_num_positives += num_pos
                
                val_pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        # --- MODIFIED: Calculate and print validation accuracy ---
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = (val_correct_cls / val_num_positives * 100) if val_num_positives > 0 else 0.0
        
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Validation Cls Acc: {val_acc:.2f}%")

    print("Training finished.")
    
    save_path = Path("yolov8_from_scratch.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path.resolve()}")


if __name__ == "__main__":
    # --- Configuration ---
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parent / "Data" / "dataset"
    classes_json_path = script_dir.parent / "Data" / "classes.json"

    IMG_SIZE = (640, 640) 
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    
    # --- Get Class Info ---
    if not classes_json_path.exists():
        print(f"Error: classes.json not found at {classes_json_path}")
        print("Please create it, e.g., {'0': 'person', '1': 'car'}")
        exit(1)
        
    with open(classes_json_path, "r") as f:
        class_map = json.load(f)
    NUM_CLASSES = len(class_map)
    print(f"Found {NUM_CLASSES} classes: {class_map.values()}")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = YOLOv8FromScratch(num_classes=NUM_CLASSES).to(device)
    
    try:
        train_loader, val_loader = get_dataloaders_for_detection(
            data_root=data_root,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
        print(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error creating DataLoaders: {e}")
        print(f"Please check that your data is at: {data_root}")
        exit(1)
        
    criterion = v8DetectionLoss(num_classes=NUM_CLASSES, reg_max=16)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Run ---
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS
    )