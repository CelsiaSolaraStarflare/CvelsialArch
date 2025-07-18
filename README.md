# CvelsialArch (CVL) Computer Vision Locator

A deep learning project that predicts GPS coordinates from images using computer vision and geographic reasoning, trained on the CVL_SFBay-1.5K Test Split dataset.

## Overview

This project implements multiple neural network architectures to predict GPS coordinates from images, with a focus on the San Francisco Bay Area. The models combine visual feature extraction with geographic reasoning to estimate location coordinates and understand terrain/scene types.

## Models
```
CVL 1.0 - Nano (Lower than 100M) - Awaiting Release
CVL 2.0 - Base (190M Params) - Awaiting Release
CVL 2.0 - High (244M Params) - Main2_High_Improved.py
CVL 2.0 - Ultra (180M Params) - Awaiting Release
CVL 2.0 - Global Ultra - main2_ultra.py
CVL 3.0 - Dynamic - Awaiting Release
CVL 3.0 - Dynamic Nano - Awaiting Release
```

## Dataset

**CVL_SFBay-1.5K Test Split**
- 1,500+ images from the San Francisco Bay Area
- GPS coordinates: Latitude [37.301, 37.900], Longitude [-122.478, -121.902]
- Includes diverse terrain types: urban, water, forest, grassland, mountain, beach
- EXIF metadata with precise GPS coordinates for training

## Performance

Models achieve meter-level accuracy on the CVL_SFBay-1.5K dataset:
- Typical errors: 800-2800 meters depending on terrain complexity
- Best performance on urban areas with distinct visual landmarks
- Challenges: Water areas, dense forests, similar-looking residential areas

## Usage

### Pretraining
If you have photos taken on any mobile device that contains EXIF you may do this:
```
python exif_extract.py
```
This code will automatically save the files to `data/exif_data.json` so that we can use it later for computations.
### Training
```bash
python main2_high_improved.py  # Train high-end model
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
streamlit run streamlit_app_high_improved.py
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

## Contributors

Trained on the CVL_SFBay-1.5K Test Split dataset `Produced by Celsia Juilyn Fan` for academic research purposes.
```
@misc{fan2025,
  author = {Fan Chengjui},
  title = {CvelsialArch for Computer Vision Locators},
  year = {2025},
  month = {jul},
  day = {17}
}
```
