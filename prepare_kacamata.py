"""
Script untuk mempersiapkan dataset kacamata untuk training YOLOv5.
Mendownload gambar dari Unsplash dan membuat anotasi menggunakan face detection.

Cara pakai:
  python prepare_kacamata.py --count 50
"""

import argparse
import os
import sys
import zipfile
import shutil
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
DATASET_DIR = ROOT / "../datasets/kacamata"


def download_images(api_key, count=30):
    """Download gambar orang pakai kacamata dari Unsplash."""
    os.makedirs(DATASET_DIR / "raw", exist_ok=True)

    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {api_key}"}
    params = {"query": "person wearing glasses face portrait", "per_page": count, "orientation": "squarish"}

    print("Downloading images from Unsplash...")
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        print(f"Unsplash API error: {resp.status_code}")
        print("Gunakan --unsplash-key KEY_ANDA atau download manual gambarnya.")
        return []

    data = resp.json()
    downloaded = []
    for i, photo in enumerate(tqdm(data.get("results", [])[:count])):
        img_url = photo["urls"]["regular"]
        img_data = requests.get(img_url).content
        fname = f"img_{i:04d}.jpg"
        (DATASET_DIR / "raw" / fname).write_bytes(img_data)
        downloaded.append(fname)

    print(f"Downloaded {len(downloaded)} images to {DATASET_DIR / 'raw'}")
    return downloaded


def create_annotations(images):
    """Buat bounding box anotasi untuk kacamata menggunakan face + eye detection."""
    train_img_dir = DATASET_DIR / "train" / "images"
    train_lbl_dir = DATASET_DIR / "train" / "labels"
    val_img_dir = DATASET_DIR / "valid" / "images"
    val_lbl_dir = DATASET_DIR / "valid" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    labeled = 0
    for i, fname in enumerate(tqdm(images)):
        img_path = DATASET_DIR / "raw" / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        has_glasses = False
        for fx, fy, fw, fh in faces:
            face_roi = gray[fy:fy + fh, fx:fx + fw]
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 4)

            if len(eyes) >= 2:
                # Assume glasses are in the upper half of the face
                eye_y_positions = [ey + fy for (ex, ey, ew, eh) in eyes]
                min_eye_y = min(eye_y_positions)
                max_eye_y = max(eye_y_positions)

                glasses_x = fx
                glasses_y = min_eye_y - int(fh * 0.15)
                glasses_w = fw
                glasses_h = int((max_eye_y - min_eye_y) * 2.5)

                glasses_y = max(0, glasses_y)
                glasses_h = min(glasses_h, h - glasses_y)

                # YOLO format: class x_center y_center width height (normalized)
                x_center = (glasses_x + glasses_w / 2) / w
                y_center = (glasses_y + glasses_h / 2) / h
                nw = glasses_w / w
                nh = glasses_h / h

                lbl_line = f"0 {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}\n"
                has_glasses = True
                break

        if has_glasses:
            is_val = i % 5 == 0  # 20% untuk validation
            if is_val:
                cv2.imwrite(str(val_img_dir / fname), img)
                with open(val_lbl_dir / f"{Path(fname).stem}.txt", "w") as f:
                    f.write(lbl_line)
            else:
                cv2.imwrite(str(train_img_dir / fname), img)
                with open(train_lbl_dir / f"{Path(fname).stem}.txt", "w") as f:
                    f.write(lbl_line)
            labeled += 1

    print(f"Successfully labeled {labeled}/{len(images)} images with glasses")
    return labeled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare kacamata dataset for YOLOv5")
    parser.add_argument("--unsplash-key", help="Unsplash API key (daftar gratis di https://unsplash.com/developers)")
    parser.add_argument("--count", type=int, default=30, help="Number of images to download")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing raw images")
    args = parser.parse_args()

    if args.skip_download:
        existing = list((DATASET_DIR / "raw").glob("*.jpg")) + list((DATASET_DIR / "raw").glob("*.png"))
        images = [f.name for f in existing]
        print(f"Found {len(images)} existing images in raw/")
    elif args.unsplash_key:
        images = download_images(args.unsplash_key, args.count)
    else:
        print("=" * 60)
        print("  PERSIAPAN DATASET KACAMATA UNTUK YOLOv5")
        print("=" * 60)
        print()
        print("  Opsi 1: Download dari Unsplash (rekomendasi)")
        print("    1. Daftar di https://unsplash.com/developers (gratis)")
        print("    2. Buat app baru, dapatkan API key")
        print("    3. Jalankan:")
        print(f"       python prepare_kacamata.py --unsplash-key KEY_ANDA --count 50")
        print()
        print("  Opsi 2: Manual")
        print("    1. Kumpulkan 30-50 gambar orang pakai kacamata")
        print("    2. Simpan di '../datasets/kacamata/raw/'")
        print("    3. Jalankan:")
        print(f"       python prepare_kacamata.py --skip-download")
        print()
        print("  Setelah dataset siap, jalankan training:")
        print("    python train.py --data data/kacamata.yaml --weights yolov5s.pt --img 640 --epochs 100 --device 0")
        print()
        sys.exit(0)

    if images:
        labeled = create_annotations(images)
        if labeled > 5:
            print()
            print("Dataset siap! Jalankan training:")
            print(f"  python train.py --data data/kacamata.yaml --weights yolov5s.pt --img 640 --epochs 100 --device 0")
    else:
        print("Tidak ada gambar untuk diproses.")
