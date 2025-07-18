import os
import json
import torch
import numpy as np
from PIL import Image
from torch import nn
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
import time
import math

# Global coordinate bounds
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0

print(f"GPS bounds: Lat [{LAT_MIN}, {LAT_MAX}], Lon [{LON_MIN}, {LON_MAX}]")

class ProperTerrainClassifier(nn.Module):
    """Dedicated terrain classifier using ResNet backbone"""
    def __init__(self):
        super().__init__()
        
        # Use ResNet50 as backbone for terrain classification
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Terrain classification head
        self.terrain_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 6),  # 6 terrain classes: urban, water, forest, grassland, mountain, beach
            nn.Softmax(dim=1)
        )
        
        # Color analysis for basic terrain detection
        self.terrain_classes = ['urban', 'water', 'forest', 'grassland', 'mountain', 'beach']
        
    def analyze_color_features(self, image_tensor):
        """Analyze color characteristics to help classify terrain"""
        # Convert to numpy for analysis (assume batch size)
        batch_size = image_tensor.shape[0]
        device = image_tensor.device
        scores = torch.zeros((batch_size, 6), device=device)
        
        for b in range(batch_size):
            img_np = image_tensor[b].permute(1, 2, 0).cpu().numpy()
            
            # Color analysis
            mean_rgb = np.mean(img_np, axis=(0, 1))
            
            # Water detection (blue dominant)
            if mean_rgb[2] > mean_rgb[0] and mean_rgb[2] > mean_rgb[1]:
                scores[b, 1] += 0.3  # water
                
            # Forest detection (green dominant)
            if mean_rgb[1] > mean_rgb[0] and mean_rgb[1] > mean_rgb[2]:
                scores[b, 2] += 0.3  # forest
                
            # Urban detection (gray/mixed colors)
            color_variance = np.var(mean_rgb)
            if color_variance < 0.05:
                scores[b, 0] += 0.2  # urban
                
            # Grassland (moderate green)
            if 0.3 < mean_rgb[1] < 0.7 and mean_rgb[1] > mean_rgb[2]:
                scores[b, 3] += 0.2  # grassland
                
            # Mountain (brown/gray tones)
            if np.all(mean_rgb > 0.3) and np.all(mean_rgb < 0.7):
                scores[b, 4] += 0.2  # mountain
                
            # Beach (light colors, sandy)
            if np.mean(mean_rgb) > 0.6:
                scores[b, 5] += 0.2  # beach
                
        return scores
        
    def forward(self, x):
        # Extract features using ResNet
        features = self.backbone(x)
        
        # Terrain classification
        terrain_logits = self.terrain_head(features)
        
        # Enhance with color analysis
        color_scores = self.analyze_color_features(x)
        
        # Combine neural network prediction with color analysis
        combined_scores = 0.8 * terrain_logits + 0.2 * color_scores
        
        # Normalize to probabilities
        terrain_probs = torch.softmax(combined_scores, dim=1)
        
        return terrain_probs

class AdvancedGPSDataset(Dataset):
    def __init__(self, image_dir, json_path, is_training=True):
        self.image_dir = image_dir
        self.is_training = is_training
        with open(json_path, 'r') as f:
            data_dict = json.load(f)

        self.image_filenames = []
        self.gps_coords = []
        self.raw_coords = []
        skipped = 0
        corrupted = 0

        # Enhanced augmentation for terrain understanding
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((576, 576)),
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # Less rotation to preserve terrain features
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        valid_coords = []
        for fname, coords in data_dict.items():
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat is not None and lon is not None:
                if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                    full_path = os.path.join(self.image_dir, fname)
                    if os.path.isfile(full_path):
                        try:
                            # Verify the image can be opened
                            with Image.open(full_path) as img:
                                img.verify()  # Verify it's a valid image
                            # Re-open to convert (verify closes the file)
                            img = Image.open(full_path).convert('RGB')
                            # If we reach here, it's valid
                            valid_coords.append((lat, lon))
                            self.image_filenames.append(full_path)
                            self.raw_coords.append([lat, lon])
                            
                            # Proper global normalization to [-1, 1] range
                            norm_lat = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * 2.0 - 1.0
                            norm_lon = (lon - LON_MIN) / (LON_MAX - LON_MIN) * 2.0 - 1.0
                            self.gps_coords.append([norm_lat, norm_lon])
                        except (IOError, SyntaxError, Exception) as e:
                            corrupted += 1
                            print(f"Corrupted image skipped: {full_path} - Error: {str(e)}")
                    else:
                        skipped += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        if valid_coords:
            lats, lons = zip(*valid_coords)
            print(f"Loaded {len(self.image_filenames)} images with GPS")
            print(f"Lat range: [{min(lats):.6f}, {max(lats):.6f}]")
            print(f"Lon range: [{min(lons):.6f}, {max(lons):.6f}]")
        
        if skipped > 0:
            print(f"Skipped {skipped} entries (missing files or invalid coords)")
        if corrupted > 0:
            print(f"Skipped {corrupted} corrupted images")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.image_filenames[idx]).convert('RGB')
        img = self.transform(img)
        norm_lat, norm_lon = self.gps_coords[idx]
        actual_lat, actual_lon = self.raw_coords[idx]
        
        return img, torch.tensor([norm_lat, norm_lon], dtype=torch.float), torch.tensor([actual_lat, actual_lon], dtype=torch.float)

class ImprovedHighEndGPSNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # GPS prediction backbone (EfficientNet for location)
        self.gps_backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get EfficientNet features
        for layer in self.gps_backbone.classifier:
            if isinstance(layer, nn.Linear):
                gps_features = layer.in_features
                break
        else:
            gps_features = 1280
            
        self.gps_backbone.classifier = nn.Identity()
        
        # Terrain classifier (separate ResNet-based) - IMPROVED VERSION
        self.terrain_classifier = ProperTerrainClassifier()
        
        # Geographic reasoning module (combines GPS and terrain features)
        self.geo_reasoner = nn.Sequential(
            nn.Linear(gps_features + 6, 512),  # GPS features + 6 terrain classes
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        # Separate coordinate prediction heads
        self.lat_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self.lon_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),  # lat_confidence, lon_confidence
            nn.Sigmoid()  # 0-1 confidence scores
        )

    def forward(self, x):
        # Extract GPS-relevant features
        gps_features = self.gps_backbone(x)
        
        # Get terrain classification (IMPROVED terrain classifier)
        terrain_probs = self.terrain_classifier(x)
        
        # Combine GPS and terrain features for geographic reasoning
        combined_input = torch.cat([gps_features, terrain_probs], dim=1)
        geo_features = self.geo_reasoner(combined_input)
        
        # Coordinate prediction
        lat = self.lat_head(geo_features)
        lon = self.lon_head(geo_features)
        coords = torch.cat([lat, lon], dim=1)
        
        # Confidence estimation
        confidence = self.confidence_head(geo_features)
        
        return coords, terrain_probs, confidence

def denormalize_coordinates(norm_lat, norm_lon):
    """Convert normalized [-1, 1] coordinates back to actual GPS coordinates"""
    lat = (norm_lat + 1.0) / 2.0 * (LAT_MAX - LAT_MIN) + LAT_MIN
    lon = (norm_lon + 1.0) / 2.0 * (LON_MAX - LON_MIN) + LON_MIN
    return lat, lon

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in meters (vectorized for batches)"""
    R = 6371000.0  # Earth radius in meters
    
    lat1_rad = lat1 * (math.pi / 180.0)
    lon1_rad = lon1 * (math.pi / 180.0)
    lat2_rad = lat2 * (math.pi / 180.0)
    lon2_rad = lon2 * (math.pi / 180.0)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    return R * c

class ImprovedAdvancedGeoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction_history = []
        
        # Loss weights
        self.coord_weight = 1.0
        self.confidence_weight = 0.3
        self.diversity_weight = 1.5  # Stronger diversity enforcement
        
    def forward(self, pred_coords, pred_terrain, pred_confidence, target_coords, actual_coords):
        batch_size = pred_coords.shape[0]
        device = pred_coords.device
        
        # Convert normalized coords back to actual coordinates
        pred_lat_batch, pred_lon_batch = denormalize_coordinates(pred_coords[:, 0], pred_coords[:, 1])
        target_lat_batch, target_lon_batch = denormalize_coordinates(target_coords[:, 0], target_coords[:, 1])
        
        # 1. COORDINATE LOSS USING HAVERSINE
        distance_error = haversine_distance(pred_lat_batch, pred_lon_batch, target_lat_batch, target_lon_batch)
        
        # 2. CONFIDENCE CALIBRATION LOSS
        confidence_lat = pred_confidence[:, 0]
        confidence_lon = pred_confidence[:, 1]
        
        # Higher confidence should mean lower error (using approximate error components)
        lat_error = torch.abs(pred_lat_batch - target_lat_batch) * 111000.0  # approx meters per degree lat
        lon_error = torch.abs(pred_lon_batch - target_lon_batch) * (111000.0 * torch.cos(target_lat_batch * (math.pi / 180.0)))  # approx for lon
        confidence_loss = torch.mean(
            confidence_lat * lat_error + confidence_lon * lon_error
        )
        
        # 3. ENHANCED DIVERSITY LOSS
        current_pred_center = [torch.mean(pred_lat_batch).item(), torch.mean(pred_lon_batch).item()]
        self.prediction_history.append(current_pred_center)
        
        if len(self.prediction_history) > 25:  # Longer history
            self.prediction_history = self.prediction_history[-25:]
        
        diversity_penalty = torch.tensor(0.0, device=device)
        if len(self.prediction_history) >= 8:  # More samples for diversity check
            recent_lats = [p[0] for p in self.prediction_history[-15:]]
            recent_lons = [p[1] for p in self.prediction_history[-15:]]
            
            lat_std = torch.tensor(recent_lats).std()
            lon_std = torch.tensor(recent_lons).std()
            
            if lat_std < 0.003 and lon_std < 0.003:  # Tighter clustering threshold
                diversity_penalty = self.diversity_weight * 3000.0  # 3km penalty for clustering
        
        # 4. COMBINE ALL LOSSES (removed geographic penalty for global dataset)
        total_loss = (
            self.coord_weight * distance_error +
            self.confidence_weight * confidence_loss +
            diversity_penalty
        )
        
        return torch.mean(total_loss)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Improved High-End GPS Model on {device}")

    image_dir = os.path.abspath('autodl-tmp/data/images_global')
    json_path = os.path.join(image_dir, 'exif_data.json')
    full_dataset = AdvancedGPSDataset(image_dir, json_path, is_training=True)
    if len(full_dataset) == 0:
        print("No images found with GPS data, please check your files.")
        return None

    # 85/15 train/val split
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    model = ImprovedHighEndGPSNet().to(device)
    
    # Different learning rates for different components
    gps_params = []
    terrain_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'gps_backbone' in name and param.requires_grad:
            gps_params.append(param)
        elif 'terrain_classifier' in name and param.requires_grad:
            terrain_params.append(param)
        elif param.requires_grad:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': gps_params, 'lr': 1e-5},      # Lower LR for pretrained GPS backbone
        {'params': terrain_params, 'lr': 2e-4},  # Medium LR for terrain classifier
        {'params': other_params, 'lr': 1e-3}     # Higher LR for new layers
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[3e-5, 5e-4, 3e-3], epochs=30, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    loss_fn = ImprovedAdvancedGeoLoss()

    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    epochs = 30

    print(f"Training with {len(train_dataset)} images, validating with {len(val_dataset)}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for imgs, target_coords, actual_coords in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs = imgs.to(device, non_blocking=True)
            target_coords = target_coords.to(device, non_blocking=True)
            actual_coords = actual_coords.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred_coords, pred_terrain, pred_confidence = model(imgs)
            loss = loss_fn(pred_coords, pred_terrain, pred_confidence, target_coords, actual_coords)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for imgs, target_coords, actual_coords in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                target_coords = target_coords.to(device, non_blocking=True)
                actual_coords = actual_coords.to(device, non_blocking=True)
                
                pred_coords, pred_terrain, pred_confidence = model(imgs)
                loss = loss_fn(pred_coords, pred_terrain, pred_confidence, target_coords, actual_coords)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1:2d} - Train: {train_loss:8.1f}m, Val: {val_loss:8.1f}m")

        # Early stopping with model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'coordinate_bounds': {
                    'lat_min': LAT_MIN, 'lat_max': LAT_MAX,
                    'lon_min': LON_MIN, 'lon_max': LON_MAX
                }
            }, 'v2-high-global.pth')
            print("New best improved model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    print("Improved High-End Training completed!")
    return model

def predict_improved_high_end(model, img_path, show_details=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_coords, pred_terrain, pred_confidence = model(x)
        
        norm_lat, norm_lon = pred_coords.squeeze().tolist()
        lat, lon = denormalize_coordinates(norm_lat, norm_lon)
        
        # Get terrain prediction
        terrain_classes = ['urban', 'water', 'forest', 'grassland', 'mountain', 'beach']
        terrain_probs = pred_terrain.squeeze()
        predicted_terrain_idx = torch.argmax(terrain_probs).item()
        predicted_terrain = terrain_classes[predicted_terrain_idx]
        terrain_confidence = terrain_probs[predicted_terrain_idx].item()
        
        # Get coordinate confidence
        lat_conf, lon_conf = pred_confidence.squeeze().tolist()
        
        if show_details:
            print(f"Predicted GPS: {lat:.6f}, {lon:.6f}")
            print(f"Predicted Terrain: {predicted_terrain} (confidence: {terrain_confidence:.3f})")
            print(f"GPS Confidence: Lat={lat_conf:.3f}, Lon={lon_conf:.3f}")
            print(f"Normalized coords: {norm_lat:.6f}, {norm_lon:.6f}")
        
        return lat, lon, predicted_terrain, terrain_confidence, (lat_conf, lon_conf)

if __name__ == '__main__':
    model = train()
    if model:
        # Load the best model
        checkpoint = torch.load('v2-high-global.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded improved model from epoch {checkpoint['epoch']} with val loss: {checkpoint['val_loss']:.1f}m")
        
        # Test on a sample image
        image_dir = os.path.abspath('autodl-tmp/data/images_global')
        json_path = os.path.join(image_dir, 'exif_data.json')
        dataset = AdvancedGPSDataset(image_dir, json_path, is_training=False)
        if len(dataset.image_filenames) > 0:
            sample_img = dataset.image_filenames[0]
            print(f"\nImproved High-End Prediction on sample: {os.path.basename(sample_img)}")
            
            # Get actual coordinates for comparison
            actual_lat, actual_lon = dataset.raw_coords[0]
            print(f"Actual GPS: {actual_lat:.6f}, {actual_lon:.6f}")
            
            result = predict_improved_high_end(model, sample_img, show_details=True)
            lat, lon, terrain, terrain_conf, coord_conf = result
            
            # Calculate error using haversine
            error_m = haversine_distance(torch.tensor([lat]), torch.tensor([lon]), torch.tensor([actual_lat]), torch.tensor([actual_lon])).item()
            print(f"Prediction error: {error_m:.1f} meters")
