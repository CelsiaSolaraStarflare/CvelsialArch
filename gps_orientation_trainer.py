import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import json
import numpy as np
from PIL import Image
import os
from datetime import datetime
import math
from typing import Tuple, Dict, Any
import torch.nn.functional as F


class GPSOrientationDataset(Dataset):
    def __init__(self, data_dir: str, exif_file: str, transform=None, input_type='rgb'):
        self.data_dir = data_dir
        self.input_type = input_type  # 'rgb' or 'depth'
        self.transform = transform

        with open(exif_file, 'r') as f:
            self.exif_data = json.load(f)

        self.image_dir = os.path.join(data_dir, 'images')
        self.depth_dir = os.path.join(data_dir, 'depths')

        # Filter out entries without valid GPS data
        self.valid_data = [item for item in self.exif_data
                           if item['gps_latitude'] is not None and item['gps_longitude'] is not None]

        # Calculate normalization stats for GPS coordinates
        lats = [item['gps_latitude'] for item in self.valid_data]
        lons = [item['gps_longitude'] for item in self.valid_data]
        self.lat_mean, self.lat_std = np.mean(lats), np.std(lats)
        self.lon_mean, self.lon_std = np.mean(lons), np.std(lons)

        print(f"Dataset initialized with {len(self.valid_data)} valid samples")
        print(f"Input type: {input_type.upper()}")
        print(f"GPS stats - Lat: {self.lat_mean:.6f}±{self.lat_std:.6f}, Lon: {self.lon_mean:.6f}±{self.lon_std:.6f}")

    def __len__(self):
        return len(self.valid_data)

    def normalize_gps(self, lat: float, lon: float) -> Tuple[float, float]:
        """Normalize GPS coordinates to have mean 0 and std 1"""
        norm_lat = (lat - self.lat_mean) / self.lat_std
        norm_lon = (lon - self.lon_mean) / self.lon_std
        return norm_lat, norm_lon

    def denormalize_gps(self, norm_lat: float, norm_lon: float) -> Tuple[float, float]:
        """Convert normalized GPS back to actual coordinates"""
        lat = norm_lat * self.lat_std + self.lat_mean
        lon = norm_lon * self.lon_std + self.lon_mean
        return lat, lon

    def parse_datetime(self, datetime_str: str) -> Tuple[float, float]:
        """Extract time of day as hour and minute normalized to [0,1]"""
        dt = datetime.fromisoformat(datetime_str)
        hour_norm = dt.hour / 23.0  # 0-23 -> 0-1
        minute_norm = dt.minute / 59.0  # 0-59 -> 0-1
        return hour_norm, minute_norm

    def encode_orientation(self, orientation: str) -> int:
        """Convert orientation string to categorical index"""
        orientation_map = {
            'Normal': 0,
            'Rotate 90 CW': 1,
            'Rotate 180': 2,
            'Rotate 270 CW': 3,
            'Flip Horizontal': 4,
            'Flip Vertical': 5
        }
        return orientation_map.get(orientation, 0)

    def __getitem__(self, idx):
        item = self.valid_data[idx]

        if self.input_type == 'rgb':
            # Load RGB image
            img_path = os.path.join(self.image_dir, item['filename'])
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            input_data = image

        elif self.input_type == 'depth':
            # Load depth data as image
            depth_filename = item['filename'].replace('.JPG', '_depth_raw.npy')
            depth_path = os.path.join(self.depth_dir, depth_filename)

            if os.path.exists(depth_path):
                depth_raw = np.load(depth_path)
                # Normalize depth values to [0, 1] range
                depth_normalized = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min())
                # Convert to PIL Image for transforms
                depth_image = Image.fromarray((depth_normalized * 255).astype(np.uint8), mode='L')

                if self.transform:
                    # Apply transforms (this will convert to tensor)
                    input_data = self.transform(depth_image)
                else:
                    # Convert to tensor manually if no transforms
                    input_data = torch.from_numpy(depth_normalized).float().unsqueeze(0)
            else:
                # Fallback: create dummy depth data
                if self.transform:
                    dummy_depth = Image.new('L', (224, 224), 0)
                    input_data = self.transform(dummy_depth)
                else:
                    input_data = torch.zeros(1, 224, 224)

        # Prepare targets
        norm_lat, norm_lon = self.normalize_gps(item['gps_latitude'], item['gps_longitude'])
        hour_norm, minute_norm = self.parse_datetime(item['datetime'])
        orientation_idx = self.encode_orientation(item['orientation'])

        targets = {
            'gps_lat': torch.tensor(norm_lat, dtype=torch.float32),
            'gps_lon': torch.tensor(norm_lon, dtype=torch.float32),
            'hour': torch.tensor(hour_norm, dtype=torch.float32),
            'minute': torch.tensor(minute_norm, dtype=torch.float32),
            'orientation': torch.tensor(orientation_idx, dtype=torch.long),
            'altitude': torch.tensor(item['gps_altitude'], dtype=torch.float32)
        }

        return {
            'input': input_data,
            'targets': targets,
            'filename': item['filename']
        }


class GPSPredictionNet(nn.Module):
    def __init__(self, input_type='rgb', num_orientation_classes=6):
        super(GPSPredictionNet, self).__init__()
        self.input_type = input_type

        # Single backbone - ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if input_type == 'rgb' else None)

        # Modify first conv layer for depth input (1 channel -> 3 channels via replication)
        if input_type == 'depth':
            # Keep the RGB pretrained weights but modify to accept 1-channel input
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # If using pretrained weights, average the RGB channels for depth
            if hasattr(original_conv1, 'weight'):
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )

        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final classification layer

        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(backbone_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Task-specific heads
        self.gps_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # lat, lon
        )

        self.time_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # hour, minute
        )

        self.orientation_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_orientation_classes)
        )

        self.altitude_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_tensor):
        # Extract features from input (RGB or depth)
        features = self.backbone(input_tensor)

        # Process features
        processed_features = self.feature_processor(features)

        # Generate predictions for each task
        gps_pred = self.gps_head(processed_features)
        time_pred = self.time_head(processed_features)
        orientation_pred = self.orientation_head(processed_features)
        altitude_pred = self.altitude_head(processed_features)

        return {
            'gps': gps_pred,  # [lat, lon] normalized
            'time': time_pred,  # [hour, minute] normalized
            'orientation': orientation_pred,  # logits for classification
            'altitude': altitude_pred  # meters
        }


class HaversineDistanceLoss(nn.Module):
    """Custom loss for GPS coordinates using Haversine distance"""

    def __init__(self, earth_radius_km=6371.0):
        super(HaversineDistanceLoss, self).__init__()
        self.earth_radius = earth_radius_km

    def forward(self, pred_coords, true_coords):
        """
        pred_coords: [batch, 2] - predicted [lat, lon] in radians
        true_coords: [batch, 2] - true [lat, lon] in radians
        """
        # Convert to radians (assuming input is in degrees)
        pred_lat, pred_lon = pred_coords[:, 0] * math.pi / 180, pred_coords[:, 1] * math.pi / 180
        true_lat, true_lon = true_coords[:, 0] * math.pi / 180, true_coords[:, 1] * math.pi / 180

        # Haversine formula
        dlat = pred_lat - true_lat
        dlon = pred_lon - true_lon

        a = torch.sin(dlat / 2) ** 2 + torch.cos(true_lat) * torch.cos(pred_lat) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        distance_km = self.earth_radius * c

        return torch.mean(distance_km)


class MultiTaskLoss(nn.Module):
    def __init__(self, dataset, weights=None):
        super(MultiTaskLoss, self).__init__()
        self.dataset = dataset
        self.haversine_loss = HaversineDistanceLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # Default weights for different tasks
        if weights is None:
            self.weights = {
                'gps': 1.0,
                'time': 0.5,
                'orientation': 0.3,
                'altitude': 0.2
            }
        else:
            self.weights = weights

    def forward(self, predictions, targets):
        # GPS loss using Haversine distance
        pred_gps = predictions['gps']
        true_gps = torch.stack([targets['gps_lat'], targets['gps_lon']], dim=1)

        # Denormalize GPS coordinates for Haversine calculation
        pred_gps_denorm = torch.zeros_like(pred_gps)
        true_gps_denorm = torch.zeros_like(true_gps)

        for i in range(pred_gps.shape[0]):
            pred_lat, pred_lon = self.dataset.denormalize_gps(
                pred_gps[i, 0].item(), pred_gps[i, 1].item()
            )
            true_lat, true_lon = self.dataset.denormalize_gps(
                true_gps[i, 0].item(), true_gps[i, 1].item()
            )
            pred_gps_denorm[i] = torch.tensor([pred_lat, pred_lon], device=pred_gps.device)
            true_gps_denorm[i] = torch.tensor([true_lat, true_lon], device=true_gps.device)

        gps_loss = self.haversine_loss(pred_gps_denorm, true_gps_denorm)

        # Time loss (MSE on normalized values)
        pred_time = predictions['time']
        true_time = torch.stack([targets['hour'], targets['minute']], dim=1)
        time_loss = self.mse_loss(pred_time, true_time)

        # Orientation loss (Cross Entropy)
        orientation_loss = self.ce_loss(predictions['orientation'], targets['orientation'])

        # Altitude loss (MSE)
        altitude_loss = self.mse_loss(predictions['altitude'].squeeze(), targets['altitude'])

        # Combine losses
        total_loss = (
                self.weights['gps'] * gps_loss +
                self.weights['time'] * time_loss +
                self.weights['orientation'] * orientation_loss +
                self.weights['altitude'] * altitude_loss
        )

        return {
            'total_loss': total_loss,
            'gps_loss': gps_loss,
            'time_loss': time_loss,
            'orientation_loss': orientation_loss,
            'altitude_loss': altitude_loss
        }


def create_data_loaders(data_dir: str, exif_file: str, batch_size=8,
                        val_split=0.2, input_type='rgb'):
    """Create training and validation data loaders"""

    # Data transforms for RGB
    if input_type == 'rgb':
        train_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Data transforms for depth (grayscale)
    elif input_type == 'depth':
        train_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            # Normalize for grayscale depth images
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # Create full dataset
    full_dataset = GPSOrientationDataset(data_dir, exif_file, transform=train_transform, input_type=input_type)

    # Split into train/val
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Apply validation transforms to validation set
    val_dataset.dataset.transform = val_transform

    # Reduce num_workers for Windows compatibility and CUDA efficiency
    num_workers = 0 if os.name == 'nt' else 2

    # Create data loaders with pin_memory for CUDA
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, full_dataset


def train_model(data_dir: str, exif_file: str, epochs=50, batch_size=8,
                learning_rate=1e-4, input_type='rgb'):
    """Main training function for single input type (RGB or depth)"""

    # Set device to CUDA by default, fall back to CPU if CUDA not available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This model requires CUDA.")
    device = torch.device('cuda')
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, dataset = create_data_loaders(
        data_dir, exif_file, batch_size, input_type=input_type
    )

    # Initialize model
    model = GPSPredictionNet(input_type=input_type).to(device)

    # Enable mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    # Initialize loss and optimizer
    criterion = MultiTaskLoss(dataset)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"Starting training for {epochs} epochs...")
    print(f"Input type: {input_type.upper()}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = batch['input'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                predictions = model(inputs)
                loss_dict = criterion(predictions, targets)
                loss = loss_dict['total_loss']

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        avg_train_loss = train_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = {k: v.to(device) for k, v in batch['targets'].items()}

                with torch.amp.autocast('cuda'):
                    predictions = model(inputs)
                    loss_dict = criterion(predictions, targets)

                val_loss_sum += loss_dict['total_loss'].item()

        avg_val_loss = val_loss_sum / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'input_type': input_type,
                'dataset_stats': {
                    'lat_mean': dataset.lat_mean,
                    'lat_std': dataset.lat_std,
                    'lon_mean': dataset.lon_mean,
                    'lon_std': dataset.lon_std
                }
            }, f'best_gps_model_{input_type}.pth')
            print(f'New best {input_type.upper()} model saved with val loss: {best_val_loss:.4f}')

    return model, train_losses, val_losses


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data"
    EXIF_FILE = "data/exif_data.json"

    print("🚀 Training RGB Model...")
    print("=" * 50)
    model_rgb, train_losses_rgb, val_losses_rgb = train_model(
        DATA_DIR, EXIF_FILE, epochs=30, batch_size=4, input_type='rgb'
    )

    print("\n🚀 Training Depth Model...")
    print("=" * 50)
    model_depth, train_losses_depth, val_losses_depth = train_model(
        DATA_DIR, EXIF_FILE, epochs=30, batch_size=4, input_type='depth'
    )

    print("\n✅ Training completed!")
    print(f"RGB model saved as: best_gps_model_rgb.pth")
    print(f"Depth model saved as: best_gps_model_depth.pth")