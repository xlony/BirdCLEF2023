import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple

class AudioPreprocessor:
    def __init__(self, sample_rate=32000, duration=5, hop_length=512, n_mels=128, fmin=20, fmax=16000):
        """
        Initialize the audio preprocessor with BirdCLEF competition parameters.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            duration (int): Duration in seconds to process from each audio file
            hop_length (int): Number of samples between successive frames
            n_mels (int): Number of mel bands
            fmin (int): Minimum frequency for mel bands
            fmax (int): Maximum frequency for mel bands
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.window_length = 1024
        self.target_length = int(duration * sample_rate)
        
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load and resample audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.ndarray: Loaded and resampled audio signal
        """
        try:
            audio, sr = sf.read(audio_path)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                
            # Handle mono/stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                
            # Handle duration
            if len(audio) > self.target_length:
                max_start = len(audio) - self.target_length
                start = np.random.randint(0, max_start)
                audio = audio[start:start + self.target_length]
            else:
                audio = np.pad(audio, (0, max(0, self.target_length - len(audio))))
                
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            return None
    
    def extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract mel spectrogram features from audio signal.
        
        Args:
            audio (np.ndarray): Audio signal
            
        Returns:
            np.ndarray: Mel spectrogram features
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                n_fft=self.window_length,
                hop_length=self.hop_length
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            return mel_spec
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

def process_dataset(input_dir: str, output_dir: str, metadata_path: str):
    """
    Process the entire dataset.
    
    Args:
        input_dir (str): Directory containing audio files
        output_dir (str): Directory to save processed features
        metadata_path (str): Path to metadata CSV file
    """
    preprocessor = AudioPreprocessor()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Process each audio file
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        audio_path = os.path.join(input_dir, row['filename'])
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue
            
        # Load and process audio
        audio = preprocessor.load_audio(audio_path)
        if audio is None:
            continue
            
        # Extract features
        features = preprocessor.extract_features(audio)
        if features is None:
            continue
            
        # Save features
        output_path = os.path.join(output_dir, f"{Path(row['filename']).stem}.npy")
        np.save(output_path, features)

if __name__ == '__main__':
    # Example usage
    process_dataset(
        input_dir='data/raw/train_audio',
        output_dir='data/processed/train',
        metadata_path='data/datasets/train_metadata.csv'
    )