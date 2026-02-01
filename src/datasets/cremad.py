"""
CREMA-D Dataset for audio-visual emotion recognition.

CREMA-D contains 7,442 video clips of actors expressing 6 emotions:
- Anger, Disgust, Fear, Happy, Neutral, Sad

Reference:
Cao et al., "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset"
IEEE Transactions on Affective Computing, 2014
"""

import os
import torch
import numpy as np
import librosa
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, Callable
import torchvision.transforms as transforms


class CREMADDataset(Dataset):
    """
    CREMA-D Dataset for audio-visual emotion recognition.

    Expected directory structure:
    root/
        AudioWAV/           # Audio files (.wav)
        VideoFlash/         # Video frames
        processedResults/   # Annotations
    """

    EMOTIONS = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

    def __init__(
        self,
        root: str,
        split: str = "train",
        visual_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 128,
    ):
        """
        Args:
            root: Root directory of CREMA-D dataset
            split: 'train' or 'test'
            visual_transform: Transform for visual input
            audio_transform: Transform for audio spectrogram
            sr: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for spectrogram
            n_mels: Number of mel bands
        """
        self.root = root
        self.split = split
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Default transforms
        if visual_transform is None:
            self.visual_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.visual_transform = visual_transform

        self.audio_transform = audio_transform

        # Load file list and labels
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load list of (audio_path, video_path, label) tuples."""
        samples = []

        audio_dir = os.path.join(self.root, "AudioWAV")
        # Look for extracted frames first, fall back to VideoFlash
        video_dir = os.path.join(self.root, "VideoFrames")
        if not os.path.exists(video_dir):
            video_dir = os.path.join(self.root, "VideoFlash")

        # List all audio files
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

        for audio_file in audio_files:
            # Parse filename: {ActorID}_{Sentence}_{Emotion}_{Level}.wav
            parts = audio_file.replace('.wav', '').split('_')
            if len(parts) >= 3:
                emotion = parts[2]
                if emotion in self.EMOTION_TO_IDX:
                    audio_path = os.path.join(audio_dir, audio_file)
                    # Video frame path (assuming extracted frames)
                    video_name = audio_file.replace('.wav', '')
                    video_path = os.path.join(video_dir, video_name)
                    label = self.EMOTION_TO_IDX[emotion]
                    samples.append((audio_path, video_path, label))

        # Split into train/test (90/10 split based on actor ID)
        # Sort by actor ID for reproducible split
        samples.sort(key=lambda x: x[0])
        n_samples = len(samples)
        n_train = int(0.9 * n_samples)

        if self.split == "train":
            samples = samples[:n_train]
        else:
            samples = samples[n_train:]

        return samples

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and convert audio to mel spectrogram."""
        try:
            # Load audio
            waveform, sr = librosa.load(audio_path, sr=self.sr)

            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )

            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize to [0, 1]
            log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)

            # Convert to tensor and add channel dimension
            spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0)

            # Resize to fixed size
            spec_tensor = torch.nn.functional.interpolate(
                spec_tensor.unsqueeze(0), size=(128, 128), mode='bilinear'
            ).squeeze(0)

            return spec_tensor

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return torch.zeros(1, 128, 128)

    def _load_visual(self, video_path: str) -> torch.Tensor:
        """Load a single frame from video."""
        try:
            # Check if path ends with .jpg (extracted frame)
            if video_path.endswith('.jpg') and os.path.exists(video_path):
                image = Image.open(video_path).convert('RGB')
            # Try to load a frame (assuming frames are extracted)
            elif os.path.exists(video_path + ".jpg"):
                image = Image.open(video_path + ".jpg").convert('RGB')
            # Try directory with frames
            elif os.path.isdir(video_path):
                frames = sorted(os.listdir(video_path))
                if frames:
                    mid_frame = frames[len(frames) // 2]
                    image = Image.open(os.path.join(video_path, mid_frame)).convert('RGB')
                else:
                    image = Image.new('RGB', (224, 224), color='black')
            else:
                image = Image.new('RGB', (224, 224), color='black')

            return self.visual_transform(image)

        except Exception as e:
            print(f"Error loading visual {video_path}: {e}")
            return torch.zeros(3, 224, 224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, video_path, label = self.samples[idx]

        audio = self._load_audio(audio_path)
        visual = self._load_visual(video_path)

        return {
            "audio": audio,
            "visual": visual,
            "label": torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.EMOTIONS)
