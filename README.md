# Preprocessing Demo Instructions

This document explains how to set up and run the **preprocess_demo.py** script to verify your preprocessing pipeline on a directory of video files.

## 1. Prerequisites

- **Python**: 3.6+
- **Dependencies**: install via pip
  ```bash
  pip install numpy opencv-python dlib tqdm
  ```
- **Dlib Shape Predictor**: download `shape_predictor_68_face_landmarks.dat` from [Dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), unzip, and place in the project root:
  ```bash
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
  mv shape_predictor_68_face_landmarks.dat /home/kvk/Documents/Ananthu_Project/
  ```

## 2. Prepare Video Folder

1. Create a directory for your videos, e.g. `videos/` inside the project:
   ```bash
   mkdir videos
   ```
2. Copy all your `.mp4`, `.avi`, `.mov` files into `videos/`.

Your project should now look like:
```
Ananthu_Project/
├── preprocess_demo.py
├── DEMO_INSTRUCTIONS.md
├── shape_predictor_68_face_landmarks.dat
└── videos/
    ├── video1.mp4
    ├── video2.avi
    └── ...
```

## 3. Run the Demo

Use the following command from the project root:

```bash
python preprocess_demo.py \
  --video_dir videos \
  --output_dir output_samples \
  --max_frames 25 \
  --img_size 224 224 \
  [--use_cache] \
  [--update_cache]
```

- `--video_dir`: path to your video folder
- `--output_dir`: directory where outputs will be saved (created if missing)
- `--max_frames`: number of frames to process per video (default: 25)
- `--img_size`: width and height for resized face patches
- `--use_cache`: load frames from cache if available
- `--update_cache`: save processed frames into cache for next runs

## 4. Output Structure

After running, inspect `output_samples/`:
```
output_samples/
└── video1/
    ├── orig_bbox/       # original frames with bounding box overlay
    ├── patches/         # cropped & resized face patches
    ├── normalized/      # NumPy `.npy` tensors (float32 [0–1])
    └── aug/             # example augmentations (flip, brightness, rotation)
```

- **Logs**: console prints show per-frame detection status, bbox coords, and padding info.
- **Cache**: processed frames are stored in `frame_cache/` to speed up future runs.

---

If you encounter errors or missing files, verify dependencies, the predictor file path, and your video directory.
