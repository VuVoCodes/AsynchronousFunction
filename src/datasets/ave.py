"""
AVE Dataset for audio-visual event localization.

AVE contains 4,143 10-second videos across 28 event classes.

Reference:
Tian et al., "Audio-Visual Event Localization in Unconstrained Videos"
ECCV 2018
"""

import os
import torch
import numpy as np
import librosa
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable
import torchvision.transforms as transforms


class AVEDataset(Dataset):
    """
    AVE Dataset for audio-visual event classification.

    Expected directory structure:
    root/
        video_frames/       # Extracted video frames
        audio/              # Audio files
        annotations/        # Train/test splits
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        visual_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        sr: int = 16000,
        n_mels: int = 128,
        num_frames: int = 3,
    ):
        """
        Args:
            root: Root directory of AVE dataset
            split: 'train', 'val', or 'test'
            visual_transform: Transform for visual input
            audio_transform: Transform for audio spectrogram
            sr: Audio sample rate
            n_mels: Number of mel bands
            num_frames: Number of frames to sample per video
        """
        self.root = root
        self.split = split
        self.sr = sr
        self.n_mels = n_mels
        self.num_frames = num_frames

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

        # Load class names and samples
        self.classes = self._load_classes()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_classes(self):
        """Load list of event classes."""
        # AVE has 28 classes - these should be loaded from dataset
        # Placeholder list
        return [
            "Church bell", "Male speech", "Bark", "Fixed-wing aircraft",
            "Race car", "Female speech", "Helicopter", "Violin",
            "Flute", "Ukulele", "Frying", "Truck", "Shofar",
            "Motorcycle", "Acoustic guitar", "Train horn", "Clock",
            "Banjo", "Goat", "Baby cry", "Bus", "Chainsaw",
            "Cat", "Horse", "Toilet flush", "Rodents", "Accordion",
            "Mandolin"
        ]

    def _load_samples(self):
        """Load list of samples for the split."""
        samples = []

        # Try to load from annotation file
        anno_file = os.path.join(self.root, "annotations", f"{self.split}.txt")

        if os.path.exists(anno_file):
            with open(anno_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        video_id = parts[0]
                        class_name = parts[1]
                        if class_name in self.class_to_idx:
                            samples.append((video_id, self.class_to_idx[class_name]))
        else:
            # Fallback: scan directories
            video_dir = os.path.join(self.root, "video_frames")
            if os.path.exists(video_dir):
                for class_name in os.listdir(video_dir):
                    class_path = os.path.join(video_dir, class_name)
                    if os.path.isdir(class_path) and class_name in self.class_to_idx:
                        for video_id in os.listdir(class_path):
                            samples.append((
                                os.path.join(class_name, video_id),
                                self.class_to_idx[class_name]
                            ))

        return samples

    def _load_audio(self, video_id: str) -> torch.Tensor:
        """Load and convert audio to mel spectrogram."""
        try:
            audio_path = os.path.join(self.root, "audio", f"{video_id}.wav")

            if not os.path.exists(audio_path):
                return torch.zeros(1, 128, 128)

            waveform, sr = librosa.load(audio_path, sr=self.sr)

            mel_spec = librosa.feature.melspectrogram(
                y=waveform, sr=sr, n_mels=self.n_mels
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)

            spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0)
            spec_tensor = torch.nn.functional.interpolate(
                spec_tensor.unsqueeze(0), size=(128, 128), mode='bilinear'
            ).squeeze(0)

            return spec_tensor

        except Exception as e:
            print(f"Error loading audio for {video_id}: {e}")
            return torch.zeros(1, 128, 128)

    def _load_visual(self, video_id: str) -> torch.Tensor:
        """Load and stack video frames."""
        try:
            frame_dir = os.path.join(self.root, "video_frames", video_id)

            if not os.path.exists(frame_dir):
                return torch.zeros(3 * self.num_frames, 224, 224)

            frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])

            if not frames:
                return torch.zeros(3 * self.num_frames, 224, 224)

            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)

            frame_tensors = []
            for idx in indices:
                frame_path = os.path.join(frame_dir, frames[idx])
                image = Image.open(frame_path).convert('RGB')
                frame_tensors.append(self.visual_transform(image))

            # Stack frames along channel dimension
            return torch.cat(frame_tensors, dim=0)

        except Exception as e:
            print(f"Error loading visual for {video_id}: {e}")
            return torch.zeros(3 * self.num_frames, 224, 224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id, label = self.samples[idx]

        audio = self._load_audio(video_id)
        visual = self._load_visual(video_id)

        return {
            "audio": audio,
            "visual": visual,
            "label": torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.classes)
