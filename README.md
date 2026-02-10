# Brand Detection & Sponsorship Valuation System

A comprehensive AI-powered system for detecting, tracking, and valuating brand logo exposure in sports videos. This project uses YOLOv11 for object detection, ByteTrack for multi-object tracking, and advanced computer vision techniques for camera angle identification and exposure quality assessment.

## 🎯 Project Overview

This system analyzes video content to:
- **Detect brand logos** using custom-trained YOLOv11 models
- **Track logo appearances** across multiple camera angles
- **Calculate media value** based on exposure duration and quality
- **Generate comprehensive reports** with financial valuations and quality metrics

### Key Features

- ✅ **Multi-Camera Detection**: Automatically identifies and tracks different camera angles using histogram-based scene fingerprinting
- ✅ **Quality Assessment**: Evaluates logo visibility using 9 quantitative variables (size, legibility, position, clutter, etc.)
- ✅ **Financial Valuation**: Calculates media value based on exposure duration and quality scores
- ✅ **Hysteresis Logic**: Prevents camera angle flickering with sticky camera detection
- ✅ **Detailed Analytics**: Generates CSV reports and executive summaries

## 📁 Project Structure

```
Brand Detection (100)/
├── logo_detector.py              # Main detection and analysis engine
├── extract_reference.py          # Utility to extract reference frames for camera angles
├── debug_similarity.py           # Debug tool to compare reference image similarity
├── best.pt                        # Custom-trained YOLOv11 model weights
├── input_video.mp4               # Input video file for analysis
├── reference_main.jpg            # Reference frame for main camera angle
├── reference_cam2.jpg            # Reference frame for secondary camera angle
├── output_annotated.mp4          # Annotated output video with detections
├── brand_exposure_report.csv     # Detailed CSV report with all metrics
└── brand_exposure_summary.txt    # Executive summary of findings
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenCV
- Ultralytics YOLOv11
- NumPy

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "/Users/bhuvanesh/Desktop/Logo Detection/Brand Detection (100)"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install required dependencies**:
   ```bash
   pip install opencv-python ultralytics numpy
   ```

### Setup

#### 1. Prepare Your Video
Place your input video as `input_video.mp4` in the project directory.

#### 2. Extract Reference Frames
Run the reference extraction utility to create camera angle fingerprints:

```bash
python extract_reference.py
```

**Interactive prompts:**
- Enter timestamp (MM:SS or seconds): `1:30` or `90`
- Enter camera name: `main` (for primary angle) or `cam2` (for secondary angle)

This creates reference images like `reference_main.jpg` and `reference_cam2.jpg`.

> **Tip**: Extract reference frames from distinct camera angles with different color profiles for better detection accuracy.

#### 3. (Optional) Verify Reference Similarity
Check if your reference images are too similar:

```bash
python debug_similarity.py
```

If similarity > 0.8, choose a different reference frame with more distinct colors.

#### 4. Run Detection
Execute the main detection script:

```bash
python logo_detector.py
```

## 📊 Output & Reports

### Generated Files

1. **`output_annotated.mp4`**
   - Annotated video with bounding boxes around detected logos
   - Color-coded: Green = counted impression, Orange = tracking
   - On-screen display: Active camera angle and similarity scores

2. **`brand_exposure_report.csv`**
   - Detailed per-impression metrics including:
     - Impression ID and Camera ID
     - Start/End timestamps and duration
     - Media value calculation
     - Quality scores (9 variables)
     - Screen share percentage

3. **`brand_exposure_summary.txt`**
   - Executive summary with:
     - Total impressions count
     - Total airtime
     - Total media value
     - Average qualitative score

### Sample Output

```
BRAND EXPOSURE ASSESSMENT: GOA
=================================

1. EXECUTIVE SUMMARY
- Total Impressions:    227
- Total Airtime:        620.50 seconds
- Total Media Value:    $1,205.78
- Avg Qualitative Score: 5.3 / 100

2. CAMERA BREAKDOWN (ROI)
Top performing cameras detailed in brand_exposure_report.csv
```

## 🔧 Technical Details

### Detection Pipeline

1. **Scene Fingerprinting**
   - Calculates HSV histogram for each frame
   - Compares against reference camera angles
   - Uses correlation matching with 0.70 threshold
   - Implements hysteresis (5% margin) to prevent flickering

2. **Logo Detection & Tracking**
   - YOLOv11 object detection on active frames
   - ByteTrack for persistent multi-object tracking
   - 0.8-second minimum duration for valid impressions

3. **Quality Assessment (9 Variables)**
   - **Size/Scale**: Logo area as percentage of screen
   - **Legibility**: Clarity based on Laplacian variance
   - **Position**: Centrality score (center = higher value)
   - **Clutter**: Edge density analysis (lower = better)
   - **Integration**: Natural vs overlay placement
   - **Obstruction**: Confidence-based visibility score
   - **Camera Movement**: Frame-to-frame IoU stability
   - **Share of Screen**: Percentage of frame occupied

4. **Financial Valuation**
   ```
   Value = (Duration / 30s) × Base Rate × (Quality Score / 100)
   ```
   - Default base rate: $100 per 30 seconds
   - Adjusted by quality score (0-100)

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reid_threshold` | 0.70 | Camera matching threshold |
| `hysteresis_margin` | 0.05 | Camera switch margin |
| `min_duration` | 0.8s | Minimum impression duration |
| `base_rate_30s` | $100 | Base valuation rate |
| `buffer_frames` | 10 | Frame buffer for scene transitions |

## 🎨 Customization

### Using a Custom Model

Replace `best.pt` with your own trained YOLOv11 model:
```bash
# Train your model
yolo train data=your_dataset.yaml model=yolo11n.pt epochs=100

# Copy weights to project
cp runs/detect/train/weights/best.pt .
```

### Adjusting Detection Sensitivity

Edit `logo_detector.py`:
```python
# Line 27: Camera matching threshold
self.reid_threshold = 0.70  # Increase for stricter matching

# Line 216: Hysteresis margin
if best_match_score < (current_cam_score + 0.05):  # Adjust margin

# Line 275: Minimum duration
if duration_sec > 0.8 and not tracks[track_id]['counted']:  # Change threshold
```

### Modifying Valuation Rates

```python
# Line 31: Base rate
self.base_rate_30s = 100.0  # Change to your desired rate
```

## 🐛 Troubleshooting

### Common Issues

**1. "No reference images found"**
- Run `extract_reference.py` first to create reference frames
- Ensure reference files are in the same directory as `logo_detector.py`

**2. Camera angles not detected correctly**
- Check reference similarity with `debug_similarity.py`
- If similarity > 0.8, choose more distinct reference frames
- Adjust `reid_threshold` in `logo_detector.py`

**3. Too many/few detections**
- Verify your YOLO model is trained on relevant logos
- Adjust confidence threshold in YOLO tracking
- Modify `min_duration` for impression counting

**4. Model not found**
- Ensure `best.pt` exists in project directory
- System falls back to `yolo11n.pt` (pretrained) if custom model missing

## 📈 Performance Metrics

- **Processing Speed**: ~15-30 FPS (depends on hardware)
- **Detection Accuracy**: Depends on custom model training
- **Camera Re-ID Accuracy**: ~95% with distinct reference frames

## 🔬 Advanced Features

### Hysteresis Logic
Prevents rapid camera switching by requiring new camera matches to be significantly better (5% margin) than the current active camera.

### Stability Scoring
Tracks frame-to-frame IoU (Intersection over Union) to measure camera movement and logo stability.

### Multi-Angle Support
Automatically detects and tracks unlimited camera angles by adding more `reference_*.jpg` files.

## 📝 License

This project is provided as-is for educational and commercial use.

## 🤝 Contributing

To improve this system:
1. Train better YOLO models with more diverse logo datasets
2. Enhance quality metrics with additional variables
3. Implement real-time processing capabilities
4. Add support for multiple brand tracking

## 📧 Support

For issues or questions, please refer to:
- YOLOv11 Documentation: https://docs.ultralytics.com/
- OpenCV Documentation: https://docs.opencv.org/

---

**Built with ❤️ using YOLOv11, OpenCV, and Python**
