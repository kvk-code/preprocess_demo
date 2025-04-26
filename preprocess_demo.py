import os
import cv2
import numpy as np
import argparse
import dlib
from tqdm import tqdm
import random
import math

# Frame cache directory
CACHE_DIR = os.path.join(os.getcwd(), 'frame_cache')

# Utility: cache path
def get_cache_path(video_path):
    base = os.path.basename(video_path)
    return os.path.join(CACHE_DIR, base + '.npz')

# Load frames from cache
def load_cache(video_path):
    path = get_cache_path(video_path)
    if os.path.exists(path):
        data = np.load(path)
        frames = data['frames']
        print(f"Loaded {frames.shape[0]} frames from cache for {video_path}")
        return frames
    return None

# Save frames to cache
def save_cache(video_path, frames):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = get_cache_path(video_path)
    np.savez_compressed(path, frames=frames)
    print(f"Saved {frames.shape[0]} frames to cache: {path}")

# Face detector setup
detector = dlib.get_frontal_face_detector()
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(shape_predictor_path)

# Detect and align face in one frame
def detect_and_align_face(frame, target_size, margin=0.2):
    dets = detector(frame, 1)
    if not dets:
        return None, None
    rect = dets[0]
    h, w = frame.shape[:2]
    mx = int((rect.right() - rect.left()) * margin)
    my = int((rect.bottom() - rect.top()) * margin)
    l = max(0, rect.left() - mx)
    t = max(0, rect.top() - my)
    r = min(w, rect.right() + mx)
    b = min(h, rect.bottom() + my)
    crop = frame[t:b, l:r]
    try:
        aligned = cv2.resize(crop, target_size)
        return aligned, (l, t, r, b)
    except Exception:
        return None, None

# Extract frames with faces, apply caching
def extract_frames_with_faces(video_path, max_frames, img_size, use_cache=False, update_cache=False):
    origs, patches, metas = [], [], []
    if use_cache:
        data = load_cache(video_path)
        if data is not None:
            frames = data
            # build patches from normalized frames
            patches = [(frame*255).astype(np.uint8) for frame in frames]
            metas = [{'idx':i,'detected':True,'bbox':None,'padded':False} for i in range(len(frames))]
            return frames, metas, patches
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        # all padding
        frames = np.zeros((max_frames,img_size[1],img_size[0],3),np.float32)
        patches = [np.zeros((img_size[1],img_size[0],3),np.uint8) for _ in range(max_frames)]
        metas = [{'idx':None,'detected':False,'bbox':None,'padded':True} for _ in range(max_frames)]
        return frames, metas, patches
    indices = np.linspace(0,total-1, min(total, max_frames*3), dtype=int)
    count=0
    for idx in indices:
        if count>=max_frames: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret: continue
        patch, bbox = detect_and_align_face(frame, img_size)
        detected = patch is not None
        if not detected:
            metas.append({'orig':frame,'idx':idx,'detected':False,'bbox':None,'padded':False})
            continue
        norm = patch.astype(np.float32)/255.0
        origs.append(frame); patches.append(patch); metas.append({'orig':frame,'idx':idx,'detected':True,'bbox':bbox,'padded':False})
        count+=1
    cap.release()
    # padding
    while count<max_frames:
        metas.append({'orig':None,'idx':None,'detected':False,'bbox':None,'padded':True})
        patches.append(np.zeros((img_size[1],img_size[0],3),np.uint8)); count+=1
    frames=np.stack([m['padded'] and np.zeros((img_size[1],img_size[0],3),np.float32) or (p.astype(np.float32)/255.0) for m,p in zip(metas,patches)],axis=0)
    if update_cache: save_cache(video_path, frames)
    print(f"Processed {video_path}: {sum(1 for m in metas if m['detected'])}/{max_frames} detected, {sum(1 for m in metas if m['padded'])} padded")
    return frames, metas, patches

# Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo preprocessing steps on videos')
    parser.add_argument('--video_dir', required=True, help='Directory with video files')
    parser.add_argument('--max_frames', type=int, default=25)
    parser.add_argument('--img_size', type=int, nargs=2, default=[224,224], help='Width Height')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--update_cache', action='store_true')
    parser.add_argument('--output_dir', default='output_samples')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    videos = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir)
              if f.lower().endswith(('.mp4','.avi','.mov'))]
    for video in videos:
        print(f"\n--- Processing {video} ---")
        frames, metas, patches = extract_frames_with_faces(video,args.max_frames,tuple(args.img_size),use_cache=args.use_cache,update_cache=args.update_cache)
        print(f"Shape: {frames.shape}")
        # prepare dirs
        vidname=os.path.splitext(os.path.basename(video))[0]
        dirs=['orig_bbox','patches','normalized','aug']
        for d in dirs: os.makedirs(os.path.join(args.output_dir,vidname,d), exist_ok=True)
        # iterate
        for i,(m,f,patch) in enumerate(zip(metas,frames,patches)):
            print(f"Frame {i}: idx={m['idx']}, detected={m['detected']}, padded={m['padded']}, bbox={m['bbox']}")
            # original with bbox or placeholder
            if 'orig' in m and m['orig'] is not None:
                img0=m['orig'].copy()
                if m['bbox']:
                    l,t,r,b=m['bbox']; cv2.rectangle(img0,(l,t),(r,b),(0,255,0),2)
                cv2.imwrite(os.path.join(args.output_dir,vidname,'orig_bbox',f"f{i}.jpg"),img0)
            # save patch
            cv2.imwrite(os.path.join(args.output_dir,vidname,'patches',f"f{i}.jpg"),patch)
            # normalized numpy
            np.save(os.path.join(args.output_dir,vidname,'normalized',f"f{i}.npy"),f)
            # augmentations
            # flip
            flip=cv2.flip(patch,1); cv2.imwrite(os.path.join(args.output_dir,vidname,'aug',f"f{i}_flip.jpg"),flip)
            # brightness
            bright=np.clip(patch*1.2+10,0,255).astype(np.uint8); cv2.imwrite(os.path.join(args.output_dir,vidname,'aug',f"f{i}_bright.jpg"),bright)
            # rotation
            M=cv2.getRotationMatrix2D((args.img_size[0]/2,args.img_size[1]/2),15,1)
            rot=cv2.warpAffine(patch,M,(args.img_size[0],args.img_size[1])); cv2.imwrite(os.path.join(args.output_dir,vidname,'aug',f"f{i}_rot.jpg"),rot)
    print("\nPreprocessing demo complete. Check output_samples/ for images and logs.")
