#!/usr/bin/env python3
"""
Download MOTChallenge datasets (MOT17, MOT20)
Usage: python download_motchallenge.py --dataset mot17 --split train --output data/MOT17
"""
import os
import sys
import argparse
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

MOT_URLS = {
    'mot17': {
        'train': 'https://motchallenge.net/data/MOT17.zip',
        'test': 'https://motchallenge.net/data/MOT17.zip',  # same zip contains both
    },
    'mot20': {
        'train': 'https://motchallenge.net/data/MOT20.zip',
        'test': 'https://motchallenge.net/data/MOT20.zip',  # same zip
    }
}

def download_file(url: str, dest: Path):
    """Download file with progress bar."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f, tqdm(
        desc=dest.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    print(f"✓ Downloaded to {dest}")

def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file with progress."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in tqdm(zf.infolist(), desc="Extracting"):
            zf.extract(member, extract_to)
    print(f"✓ Extracted to {extract_to}")

def verify_mot_structure(base_path: Path, dataset: str, split: str):
    """Verify the downloaded dataset structure."""
    expected_folder = base_path / dataset.upper() / split
    
    if not expected_folder.exists():
        print(f"⚠ Warning: Expected folder not found: {expected_folder}")
        return False
    
    sequences = [d for d in expected_folder.iterdir() if d.is_dir()]
    if not sequences:
        print(f"⚠ Warning: No sequences found in {expected_folder}")
        return False
    
    print(f"\n✓ Found {len(sequences)} sequences in {expected_folder}:")
    for seq in sorted(sequences):
        img_folder = seq / "img1"
        gt_file = seq / "gt" / "gt.txt"
        seqinfo = seq / "seqinfo.ini"
        
        img_count = len(list(img_folder.glob("*.jpg"))) if img_folder.exists() else 0
        has_gt = "✓" if gt_file.exists() else "✗"
        has_info = "✓" if seqinfo.exists() else "✗"
        
        print(f"  • {seq.name}: {img_count} images | GT: {has_gt} | SeqInfo: {has_info}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download MOTChallenge datasets")
    parser.add_argument('--dataset', required=True, choices=['mot17', 'mot20'],
                       help="Which dataset to download")
    parser.add_argument('--split', default='train', choices=['train', 'test', 'both'],
                       help="Which split to download")
    parser.add_argument('--output', type=str, default='data',
                       help="Output directory (will create MOT17/MOT20 subdirs)")
    parser.add_argument('--skip-extract', action='store_true',
                       help="Skip extraction if zip already exists")
    parser.add_argument('--keep-zip', action='store_true',
                       help="Keep zip file after extraction")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_upper = args.dataset.upper()
    zip_filename = f"{dataset_upper}.zip"
    zip_path = output_path / zip_filename
    
    # Download if zip doesn't exist
    if not zip_path.exists():
        url = MOT_URLS[args.dataset]['train']  # same zip for train/test
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"✗ Download failed: {e}")
            print("\n💡 Manual download instructions:")
            print(f"   1. Visit: https://motchallenge.net/data/{dataset_upper}/")
            print(f"   2. Download {dataset_upper}.zip")
            print(f"   3. Place it in: {output_path.absolute()}")
            sys.exit(1)
    else:
        print(f"✓ Zip file already exists: {zip_path}")
    
    # Extract
    if not args.skip_extract:
        extract_zip(zip_path, output_path)
        
        # Verify structure
        splits_to_verify = ['train', 'test'] if args.split == 'both' else [args.split]
        for split in splits_to_verify:
            if not verify_mot_structure(output_path, args.dataset, split):
                print(f"\n⚠ Structure verification failed for {split} split")
    
    # Cleanup
    if not args.keep_zip and not args.skip_extract:
        print(f"\nRemoving zip file: {zip_path}")
        zip_path.unlink()
    
    print("\n" + "="*60)
    print("✓ Setup complete!")
    print("="*60)
    print(f"\nDataset location: {output_path / dataset_upper}")
    print("\nNext steps:")
    print(f"  1. Verify your data structure:")
    print(f"     ls -la {output_path / dataset_upper / 'train'}/")
    print(f"\n  2. Run tracking evaluation:")
    print(f"     python eval/run_mot.py \\")
    print(f"       --dataset {args.dataset} \\")
    print(f"       --split train \\")
    print(f"       --data_root {output_path / dataset_upper}")

if __name__ == '__main__':
    main()