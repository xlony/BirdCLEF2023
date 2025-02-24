import os
import kaggle
from pathlib import Path

def download_birdclef2023_dataset(data_dir="data"):
    """
    Download and extract audio files from the BirdCLEF 2023 dataset from Kaggle.
    Preserves existing CSV files in the datasets directory.
    
    Args:
        data_dir (str): Directory to save the dataset files
    """
    # Create data directory if it doesn't exist
    raw_data_dir = Path(data_dir) / "raw"
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Download dataset using Kaggle API
    try:
        print("Authenticating with Kaggle API...")
        kaggle.api.authenticate()
        
        print("Downloading dataset files...")
        kaggle.api.competition_download_files(
            'birdclef-2023',
            path=raw_data_dir
        )
        print(f"Dataset downloaded successfully to {raw_data_dir}")
        
        # Extract files
        zip_path = raw_data_dir / "birdclef-2023.zip"
        if zip_path.exists():
            print("Extracting audio files...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of audio files
                audio_files = [f for f in zip_ref.namelist() if f.endswith('.ogg') or f.endswith('.wav')]
                
                # Extract only audio files
                for audio_file in audio_files:
                    zip_ref.extract(audio_file, raw_data_dir)
                print(f"Extracted {len(audio_files)} audio files")
                
                # Remove the zip file to save space
                os.remove(zip_path)
                print("Removed downloaded zip file to save space")
        else:
            print("Warning: Downloaded zip file not found")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Installed kaggle package: pip install kaggle")
        print("2. Created a Kaggle account and downloaded API credentials")
        print("3. Placed kaggle.json in ~/.kaggle/ directory")
        print("4. Have proper permissions for the data directory")
        raise

if __name__ == "__main__":
    download_birdclef2023_dataset()