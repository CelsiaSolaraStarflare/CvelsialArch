import os
import json
import torch
import folium
import numpy as np
import requests
from PIL import Image
from torch import nn
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
import time

# Actual coordinate bounds from the SF Bay Area dataset analysis
LAT_MIN, LAT_MAX = 37.301, 37.900  # With 5% padding
LON_MIN, LON_MAX = -122.478, -121.902  # With 5% padding

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

class FoliumTerrainValidator:
    """Enhanced geographic validator with better SF Bay Area mapping"""
    def __init__(self):
        # More precise SF Bay Area geographic features
        self.water_areas = {
            'pacific_ocean': {'lat_range': (37.3, 37.9), 'lon_range': (-122.52, -122.46)},
            'sf_bay': {'lat_range': (37.4, 37.85), 'lon_range': (-122.45, -122.2)},
            'richardson_bay': {'lat_range': (37.85, 37.89), 'lon_range': (-122.52, -122.48)}
        }
        
        self.urban_areas = {
            'sf_downtown': {'lat_range': (37.77, 37.82), 'lon_range': (-122.42, -122.38)},
            'mission': {'lat_range': (37.74, 37.77), 'lon_range': (-122.42, -122.40)},
            'oakland_downtown': {'lat_range': (37.79, 37.83), 'lon_range': (-122.28, -122.26)},
            'berkeley': {'lat_range': (37.86, 37.88), 'lon_range': (-122.27, -122.25)},
            'south_bay_urban': {'lat_range': (37.33, 37.45), 'lon_range': (-122.1, -121.95)}
        }
        
        self.forest_areas = {
            'golden_gate_park': {'lat_range': (37.76, 37.77), 'lon_range': (-122.51, -122.45)},
            'presidio': {'lat_range': (37.79, 37.81), 'lon_range': (-122.47, -122.44)},
            'marin_headlands': {'lat_range': (37.82, 37.87), 'lon_range': (-122.52, -122.47)},
            'tilden_park': {'lat_range': (37.89, 37.92), 'lon_range': (-122.25, -122.22)}
        }
        
        self.mountain_areas = {
            'twin_peaks': {'lat_range': (37.75, 37.76), 'lon_range': (-122.45, -122.44)},
            'mount_davidson': {'lat_range': (37.73, 37.74), 'lon_range': (-122.45, -122.44)},
            'hills_peninsula': {'lat_range': (37.35, 37.5), 'lon_range': (-122.35, -122.2)}
        }
        
        self.beach_areas = {
            'ocean_beach': {'lat_range': (37.72, 37.78), 'lon_range': (-122.51, -122.50)},
            'baker_beach': {'lat_range': (37.79, 37.80), 'lon_range': (-122.48, -122.47)},
            'crissy_field': {'lat_range': (37.80, 37.81), 'lon_range': (-122.47, -122.45)}
        }
        
    def get_actual_terrain_at_location(self, lat, lon):
        """Get actual terrain type at coordinates using detailed geographic mapping"""
        
        # Check water areas first (highest priority)
        for area_name, bounds in self.water_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'water'
        
        # Check beach areas
        for area_name, bounds in self.beach_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'beach'
        
        # Check urban areas
        for area_name, bounds in self.urban_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'urban'
        
        # Check forest areas  
        for area_name, bounds in self.forest_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'forest'
                
        # Check mountain areas
        for area_name, bounds in self.mountain_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'mountain'
        
        # Default classification based on general SF geography
        if lon < -122.42:  # Western SF - more residential/mixed
            return 'grassland'  # Residential areas with grass/parks
        elif 37.45 < lat < 37.7 and -122.2 < lon < -121.95:  # South Bay
            return 'urban'
        else:
            return 'grassland'  # Default to grassland for residential areas
    
    def validate_prediction(self, predicted_terrain, actual_lat, actual_lon):
        """Check if predicted terrain matches actual location"""
        actual_terrain = self.get_actual_terrain_at_location(actual_lat, actual_lon)
        
        # Define incompatible combinations (high penalty)
        incompatible = {
            'water': ['urban', 'mountain', 'grassland', 'forest'],
            'urban': ['water', 'beach'],
            'mountain': ['water', 'beach'],
            'beach': ['urban', 'mountain', 'forest']
        }
        
        # Check for major incompatibilities
        if actual_terrain in incompatible:
            if predicted_terrain in incompatible[actual_terrain]:
                return 'incompatible'  # Heavy penalty
        
        # Check for exact matches (reward)
        if predicted_terrain == actual_terrain:
            return 'exact_match'  # Reward
            
        # Partial matches (neutral)
        partial_matches = {
            'grassland': ['urban', 'forest'],
            'urban': ['grassland'],
            'forest': ['grassland', 'mountain'],
            'mountain': ['forest', 'grassland']
        }
        
        if actual_terrain in partial_matches:
            if predicted_terrain in partial_matches[actual_terrain]:
                return 'partial_match'
        
        return 'mismatch'  # Small penalty

class AdvancedGPSDataset(Dataset):
    def __init__(self, image_dir, json_path, is_training=True):
        self.image_dir = image_dir
        self.is_training = is_training
        with open(json_path, 'r') as f:
            data_list = json.load(f)

        self.image_filenames = []
        self.gps_coords = []
        self.raw_coords = []
        skipped = 0

        # Enhanced augmentation for terrain understanding
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((288, 288)),
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # Less rotation to preserve terrain features
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        valid_coords = []
        for entry in data_list:
            fname = entry.get("filename")
            lat = entry.get("gps_latitude")
            lon = entry.get("gps_longitude")
            if (fname and lat is not None and lon is not None and
                fname.lower().endswith(('.jpg', '.jpeg', '.png'))):
                if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                    valid_coords.append((lat, lon))
                    full_path = os.path.join(self.image_dir, fname)
                    if os.path.isfile(full_path):
                        self.image_filenames.append(full_path)
                        self.raw_coords.append([lat, lon])
                        
                        # Proper regional normalization to [-1, 1] range
                        norm_lat = 2.0 * (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) - 1.0
                        norm_lon = 2.0 * (lon - LON_MIN) / (LON_MAX - LON_MIN) - 1.0
                        self.gps_coords.append([norm_lat, norm_lon])
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
            print(f"Skipped {skipped} entries")

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

class ImprovedAdvancedGeoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo_validator = FoliumTerrainValidator()
        self.prediction_history = []
        
        # Loss weights
        self.coord_weight = 1.0
        self.terrain_weight = 0.8  # Increased for better terrain learning
        self.geographic_weight = 3.0  # Higher penalty for geographic impossibilities
        self.confidence_weight = 0.3
        self.diversity_weight = 1.5  # Stronger diversity enforcement
        
    def forward(self, pred_coords, pred_terrain, pred_confidence, target_coords, actual_coords):
        batch_size = pred_coords.shape[0]
        device = pred_coords.device
        
        # Convert normalized coords back to actual coordinates
        pred_lat_batch = (pred_coords[:, 0] + 1.0) / 2.0 * (LAT_MAX - LAT_MIN) + LAT_MIN
        pred_lon_batch = (pred_coords[:, 1] + 1.0) / 2.0 * (LON_MAX - LON_MIN) + LON_MIN
        target_lat_batch = (target_coords[:, 0] + 1.0) / 2.0 * (LAT_MAX - LAT_MIN) + LAT_MIN
        target_lon_batch = (target_coords[:, 1] + 1.0) / 2.0 * (LON_MAX - LON_MIN) + LON_MIN
        
        # 1. COORDINATE LOSS
        lat_diff = torch.abs(pred_lat_batch - target_lat_batch)
        lon_diff = torch.abs(pred_lon_batch - target_lon_batch)
        
        lat_to_meters = 111000
        lon_to_meters = 111000 * torch.cos(torch.tensor(37.5 * torch.pi / 180))
        
        lat_error_m = lat_diff * lat_to_meters
        lon_error_m = lon_diff * lon_to_meters
        distance_error = torch.sqrt(lat_error_m**2 + lon_error_m**2)
        
        # 2. IMPROVED GEOGRAPHIC CONSISTENCY LOSS
        geographic_penalty = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            # Get predicted terrain (dominant class)
            terrain_idx = torch.argmax(pred_terrain[i]).item()
            terrain_classes = ['urban', 'water', 'forest', 'grassland', 'mountain', 'beach']
            predicted_terrain = terrain_classes[terrain_idx]
            
            # Get actual location
            actual_lat = actual_coords[i][0].item()
            actual_lon = actual_coords[i][1].item()
            
            # Validate geographic consistency using improved validator
            validation_result = self.geo_validator.validate_prediction(predicted_terrain, actual_lat, actual_lon)
            
            if validation_result == 'incompatible':
                geographic_penalty[i] = 8000.0  # 8km penalty for impossible predictions
            elif validation_result == 'exact_match':
                geographic_penalty[i] = -1000.0   # 1km reward for correct terrain
            elif validation_result == 'partial_match':
                geographic_penalty[i] = 0.0       # Neutral
            else:  # mismatch
                geographic_penalty[i] = 2000.0    # 2km penalty for terrain mismatch
        
        # 3. CONFIDENCE CALIBRATION LOSS
        confidence_lat = pred_confidence[:, 0]
        confidence_lon = pred_confidence[:, 1]
        
        # Higher confidence should mean lower error
        confidence_loss = torch.mean(
            confidence_lat * lat_error_m + confidence_lon * lon_error_m
        )
        
        # 4. ENHANCED DIVERSITY LOSS
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
        
        # 5. COMBINE ALL LOSSES
        total_loss = (
            self.coord_weight * distance_error +
            self.geographic_weight * geographic_penalty +
            self.confidence_weight * confidence_loss +
            diversity_penalty
        )
        
        return torch.mean(total_loss)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Improved High-End GPS Model on {device}")

    full_dataset = AdvancedGPSDataset('data/images', 'data/exif_data.json', is_training=True)
    if len(full_dataset) == 0:
        print("No images found with GPS data, please check your files.")
        return None

    # 85/15 train/val split
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

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
            }, 'improved_high_end_gps_model.pth')
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
        transforms.Resize((256, 256)),
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
            
            # Validate against known geography
            geo_validator = FoliumTerrainValidator()
            actual_terrain = geo_validator.get_actual_terrain_at_location(lat, lon)
            validation = geo_validator.validate_prediction(predicted_terrain, lat, lon)
            
            print(f"Expected terrain at location: {actual_terrain}")
            print(f"Geographic validation: {validation}")
        
        return lat, lon, predicted_terrain, terrain_confidence, (lat_conf, lon_conf)

if __name__ == '__main__':
    model = train()
    if model:
        # Load the best model
        checkpoint = torch.load('improved_high_end_gps_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded improved model from epoch {checkpoint['epoch']} with val loss: {checkpoint['val_loss']:.1f}m")
        
        # Test on a sample image
        dataset = AdvancedGPSDataset('data/images', 'data/exif_data.json', is_training=False)
        if len(dataset.image_filenames) > 0:
            sample_img = dataset.image_filenames[0]
            print(f"\nImproved High-End Prediction on sample: {os.path.basename(sample_img)}")
            
            # Get actual coordinates for comparison
            actual_lat, actual_lon = dataset.raw_coords[0]
            print(f"Actual GPS: {actual_lat:.6f}, {actual_lon:.6f}")
            
            result = predict_improved_high_end(model, sample_img, show_details=True)
            lat, lon, terrain, terrain_conf, coord_conf = result
            
            # Calculate error
            error_lat = abs(lat - actual_lat)
            error_lon = abs(lon - actual_lon)
            error_m = ((error_lat * 111000)**2 + (error_lon * 89000)**2)**0.5
            print(f"Prediction error: {error_m:.1f} meters")