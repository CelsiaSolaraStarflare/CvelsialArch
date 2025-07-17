# GPS Localization from Images

A deep learning project that predicts GPS coordinates from images using computer vision and geographic reasoning, trained on the CVL_SFBay-1.5K Test Split dataset.

## Overview

This project implements multiple neural network architectures to predict GPS coordinates from images, with a focus on the San Francisco Bay Area. The models combine visual feature extraction with geographic reasoning to estimate location coordinates and understand terrain/scene types.

## Dataset

**CVL_SFBay-1.5K Test Split**
- 1,500+ images from the San Francisco Bay Area
- GPS coordinates: Latitude [37.301, 37.900], Longitude [-122.478, -121.902]
- Includes diverse terrain types: urban, water, forest, grassland, mountain, beach
- EXIF metadata with precise GPS coordinates for training

## Model Architectures

### 1. High-End GPS Model (`main2_high_improved.py`)
- **Parameters**: 29.5 million
- **Architecture**: EfficientNet-B0 + ResNet50 dual backbone
- **Features**: 
  - GPS coordinate prediction
  - Terrain classification (6 classes)
  - Geographic consistency validation
  - Confidence estimation
- **Use case**: High accuracy predictions with detailed terrain analysis

### 2. Ultra GPS Model (`main2_ultra.py`)
- **Parameters**: 16.6 million  
- **Architecture**: EfficientNet-B0 + ResNet18 dual backbone
- **Features**:
  - GPS coordinate prediction
  - Indoor/outdoor classification
  - Landscape type classification (6 classes)
  - Indoor scene classification (8 classes)
  - Object detection (10 classes)
  - Real-time geographic validation via OpenStreetMap API
  - Adaptive sampling system
- **Use case**: Comprehensive scene understanding with multi-task learning

### 3. Other Variants
- `main2_accurate.py`: Accuracy-focused model
- `main2_fixed.py`: Bug-fixed baseline
- `streamlit_app.py`: Interactive web interface

## Key Features

- **Multi-modal Learning**: Combines visual features with geographic reasoning
- **Terrain-aware Predictions**: Models understand landscape types and geographic constraints
- **Confidence Estimation**: Provides uncertainty estimates for predictions
- **Interactive Visualization**: Folium-based map visualization of predictions
- **Geographic Validation**: Real-time validation against OpenStreetMap data

## Training Process

1. **Data Preprocessing**: Images resized to 256x256, normalized for ImageNet
2. **Augmentation**: Random crops, flips, rotation, color jittering
3. **Loss Function**: Custom geographic loss combining:
   - Coordinate distance error
   - Terrain classification loss
   - Geographic consistency penalties
   - Confidence calibration
   - Diversity enforcement

4. **Optimization**: AdamW with different learning rates for pretrained vs. new layers
5. **Validation**: 85/15 train/validation split with early stopping

## Performance

Models achieve meter-level accuracy on the CVL_SFBay-1.5K dataset:
- Typical errors: 100-2000 meters depending on terrain complexity
- Best performance on urban areas with distinct visual landmarks
- Challenges: Water areas, dense forests, similar-looking residential areas

## Usage

### Training
```bash
python main2_high_improved.py  # Train high-end model
python main2_ultra.py          # Train ultra model
```

### Inference
```bash
python -c "
from main2_high_improved import predict_improved_high_end, ImprovedHighEndGPSNet
import torch

# Load trained model
model = ImprovedHighEndGPSNet()
checkpoint = torch.load('improved_high_end_gps_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict location
lat, lon, terrain, terrain_conf, coord_conf = predict_improved_high_end(model, 'path/to/image.jpg')
print(f'Predicted location: {lat:.6f}, {lon:.6f}')
print(f'Terrain: {terrain} (confidence: {terrain_conf:.3f})')
"
```

### Web Interface
```bash
streamlit run streamlit_app.py
```

## File Structure

```
├── main2_high_improved.py      # High-end GPS model (29.5M params)
├── main2_ultra.py             # Ultra GPS model (16.6M params)
├── main2_accurate.py          # Accuracy-focused variant
├── streamlit_app.py           # Web interface
├── data/
│   ├── images/               # Training images
│   ├── exif_data.json       # GPS metadata
│   └── location_data.csv    # Location dataset
├── *.pth                     # Trained model weights
└── README.md                # This file
```

## Requirements

```
torch
torchvision
PIL
folium
numpy
streamlit
requests
tqdm
```

## Geographic Scope

Currently optimized for the San Francisco Bay Area, including:
- San Francisco downtown and neighborhoods
- Oakland and Berkeley
- Marin County
- South Bay urban areas
- Pacific Ocean coastline
- Bay Area mountains and forests

## Future Improvements

- Expand to other geographic regions
- Incorporate temporal information (season, time of day)
- Add elevation prediction
- Improve water body detection
- Optimize inference speed
- Mobile deployment

## Contributors

Trained on the CVL_SFBay-1.5K Test Split dataset for academic research purposes.