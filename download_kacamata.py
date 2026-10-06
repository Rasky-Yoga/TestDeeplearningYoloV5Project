"""
Download dataset kacamata dari Roboflow Universe untuk training YOLOv5.

Cara pakai:
1. Buka https://universe.roboflow.com/diplom1/yolov8-glasses
2. Klik "Download Dataset" -> pilih format "YOLOv5"
3. Copy link "curl" yang diberikan, atau dapatkan API key dari Roboflow

Alternatif dataset lain:
- https://universe.roboflow.com/glasses-or-no-glasses/yolo-sample-train-hskj5 (72 gambar, 1 kelas: Glasses)
- https://universe.roboflow.com/yolo-practice-xgsmc/glasses-iy1og (2.7k gambar, multi kelas)
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

DATASETS = {
    "glasses": {
        "url": "https://universe.roboflow.com/ds/...",  # ganti dengan URL dari Roboflow
        "path": ROOT / "../datasets/kacamata",
    },
}


def download_roboflow(api_key, workspace, project, version, output_dir):
    """Download dataset dari Roboflow menggunakan API."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Menginstall roboflow package...")
        os.system(f"{sys.executable} -m pip install roboflow")

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov5", location=output_dir)
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def setup_yaml(dataset_path):
    """Buat data YAML untuk training."""
    yaml_content = f"""path: {dataset_path}
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: kacamata
"""
    yaml_path = Path(dataset_path) / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"YAML config created: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download kacamata dataset from Roboflow")
    parser.add_argument("--api-key", help="Roboflow API key (dapatkan di https://app.roboflow.com/settings/api)")
    parser.add_argument("--workspace", default="diplom1", help="Roboflow workspace name")
    parser.add_argument("--project", default="yolov8-glasses", help="Roboflow project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number")
    parser.add_argument("--output", default=ROOT / "../datasets/kacamata", help="Output directory")

    args = parser.parse_args()

    if not args.api_key:
        print("=" * 60)
        print("  CARA DOWNLOAD DATASET KACAMATA UNTUK TRAINING")
        print("=" * 60)
        print()
        print("  1. Daftar akun gratis di https://app.roboflow.com")
        print("  2. Buka dataset: https://universe.roboflow.com/diplom1/yolov8-glasses")
        print("  3. Klik 'Download Dataset' -> pilih format 'YOLOv5'")
        print("  4. Salin API key dari halaman settings API")
        print()
        print("  Jalankan lagi dengan --api-key <KEY_KAMU>:")
        print(f"    python download_kacamata.py --api-key KEY_ANDA")
        print()
        print("  Dataset rekomendasi (1 kelas: kacamata):")
        print("    https://universe.roboflow.com/glasses-or-no-glasses/yolo-sample-train-hskj5")
        print()
        print("  Dataset alternatif (multi kelas):")
        print("    https://universe.roboflow.com/yolo-practice-xgsmc/glasses-iy1og")
        print()
        print("  ATAU download manual, extract ke '../datasets/kacamata/', lalu jalankan:")
        print(f"    python train.py --data data/kacamata.yaml --weights yolov5s.pt --img 640 --epochs 100")
        print()
    else:
        dataset_path = download_roboflow(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=str(args.output),
        )
        setup_yaml(dataset_path)
        print()
        print("Dataset siap! Jalankan training:")
        print(f"  python train.py --data {Path(dataset_path) / 'dataset.yaml'} --weights yolov5s.pt --img 640 --epochs 100 --device 0")
