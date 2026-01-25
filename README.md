# Face Recognition System

A real-time face recognition system using **ArcFace embeddings** and **Haar Cascades** with **FaceMesh landmarks** for robust face alignment. This project provides enrollment and recognition pipelines optimized for CPU performance.

## Project Overview

This system implements a complete face recognition workflow:
1. **Face Detection**: Using Haar Cascade classifiers to locate faces
2. **Facial Landmark Detection**: Using MediaPipe FaceMesh (5-point landmarks)
3. **Face Alignment**: Aligning faces to a standard 112x112 size
4. **Embedding Generation**: Using ArcFace ONNX model for creating embeddings
5. **Face Recognition**: Matching detected faces against a known database using cosine distance

## Project Structure

```
Face-Recognition/
├── init_project.py          # Initialize project directory structure
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/
│   ├── enroll/            # Raw enrolled face images (organized by identity)
│   └── db/                # Generated database files
├── models/
│   └── embedder_arcface.onnx  # Pre-trained ArcFace embedding model
└── src/
    ├── camera.py          # Camera utilities
    ├── detect.py          # Face detection pipeline
    ├── landmarks.py       # Facial landmark detection
    ├── align.py           # Face alignment utilities
    ├── embed.py           # Embedding generation
    ├── enroll.py          # Enrollment tool (add new faces to DB)
    ├── recognize.py       # Real-time recognition script
    ├── evaluate.py        # Model evaluation utilities
    └── haar_5pt.py        # Haar cascade + 5-point alignment implementation
```

## Installation

### 1. Setup Project Structure
```bash
python init_project.py
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Requirements:**
- opencv-python - Image processing
- numpy - Numerical computations
- onnxruntime - Running the ArcFace model
- scipy - Scientific computing
- tqdm - Progress bars
- mediapipe - Facial landmark detection

### 3. Download Pre-trained Model
Ensure `embedder_arcface.onnx` is placed in the `models/` directory. This is the ArcFace embedding model used for generating face embeddings.

## Usage

### Enroll New Faces
Add new identities to the face database:

```bash
python -m src.enroll
```

**Controls:**
- **SPACE**: Capture a sample (when face is detected)
- **a**: Toggle auto-capture mode
- **s**: Save enrollment (after capturing enough samples)
- **r**: Reset captured samples (keeps existing crops)
- **q**: Quit

**Output:**
- `data/db/face_db.npz` - Face embeddings database
- `data/db/face_db.json` - Metadata
- `data/enroll/<name>/` - Aligned face crops per identity

### Recognize Faces
Real-time face recognition from camera:

```bash
python -m src.recognize
```

**Controls:**
- **q**: Quit
- **r**: Reload database from disk
- **+/-**: Adjust recognition threshold
- **d**: Toggle debug overlay

**Features:**
- Multi-face recognition
- Real-time cosine distance matching
- Adjustable recognition threshold
- Debug visualization mode

### Run Embedding Pipeline (Development)
```bash
python -m src.embed
```

**Controls:**
- **q**: Quit
- **p**: Print embedding statistics

### Evaluate Model
```bash
python -m src.evaluate
```

## Technical Details

### Pipeline Architecture

```
Camera Input
    ↓
Haar Cascade Detection (Multi-face)
    ↓
Per-face ROI Processing
    ├→ MediaPipe FaceMesh (5-point landmarks)
    ├→ Face Alignment (112x112)
    ├→ ArcFace Embedding (ONNX)
    ↓
Cosine Distance Matching
    ↓
Recognition Output
```

### Face Alignment
- Uses 5-point landmarks (eyes, nose, mouth)
- Aligns faces to canonical 112x112 image
- Handles rotation and scale normalization

### Embeddings
- Generated using ArcFace ONNX model
- L2-normalized 512-dimensional vectors
- Stored as mean embeddings per identity

### Recognition Matching
- Cosine distance metric: `distance = 1 - cosine_similarity`
- Configurable threshold for acceptance
- Supports multi-face detection and labeling

## Key Features

✓ **Real-time Processing**: CPU-optimized with ONNX runtime  
✓ **Multi-face Detection**: Detects and recognizes multiple faces simultaneously  
✓ **Robust Alignment**: 5-point landmark-based alignment  
✓ **Interactive Controls**: Live threshold adjustment during recognition  
✓ **Database Management**: Store and manage multiple identities  
✓ **Debug Visualization**: Optional overlay for development  

## Performance Notes

- Optimized for CPU inference using ONNX Runtime
- FaceMesh landmarks processed per Haar ROI (avoids inconsistency)
- Suitable for real-time embedded applications
- Adjustable recognition threshold for precision/recall tradeoff

## Data Organization

**Enrollment Phase:**
- Captured faces stored in `data/enroll/<identity_name>/` as aligned 112x112 crops
- Multiple samples per identity improve embedding robustness

**Database:**
- `face_db.npz` - Serialized embeddings
- `face_db.json` - Identity metadata and statistics

## Troubleshooting

- **No faces detected**: Ensure good lighting and camera angle
- **False positives**: Increase recognition threshold
- **Poor alignment**: Ensure facial landmarks are clear and well-lit
- **Slow performance**: Verify ONNX Runtime is using GPU if available

## License

Academic project for Year 3 Robotics course

---

**Last Updated**: January 2026

Made with 💓 by Joyeuse 