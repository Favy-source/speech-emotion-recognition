#!/usr/bin/env python3
"""
Dataset download script for Speech Emotion Recognition project.
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, filename: str) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def download_datasets():
    """Download required datasets."""
    print("📊 Starting dataset downloads...")
    
    # Create data directories
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "RAVDESS": {
            "url": "https://example.com/ravdess.zip",  # Replace with actual URL
            "description": "Ryerson Audio-Visual Database of Emotional Speech and Song"
        },
        "TESS": {
            "url": "https://example.com/tess.zip",  # Replace with actual URL  
            "description": "Toronto Emotional Speech Set"
        }
    }
    
    print("⚠️  Manual download required!")
    print("\nPlease manually download the following datasets:")
    print("\n1. RAVDESS Dataset:")
    print("   - Visit: https://zenodo.org/record/1188976")
    print("   - Download and extract to: data/raw/ravdess/")
    print("\n2. TESS Dataset:")
    print("   - Visit: https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF")
    print("   - Download and extract to: data/raw/tess/")
    print("\n3. FER+ Dataset (optional):")
    print("   - Visit: https://github.com/Microsoft/FERPlus")
    print("   - Download and extract to: data/raw/fer_plus/")
    
    # Create sample structure
    for dataset in ["ravdess", "tess", "fer_plus"]:
        dataset_dir = data_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        (dataset_dir / "README.txt").write_text(f"Download {dataset.upper()} dataset here")
    
    print("\n✅ Created dataset directory structure")
    print("📥 Please download datasets manually and place them in the appropriate directories")

if __name__ == "__main__":
    download_datasets()
