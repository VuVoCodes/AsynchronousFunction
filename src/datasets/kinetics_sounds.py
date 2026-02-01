"""
Kinetics-Sounds Dataset for audio-visual action recognition.

Kinetics-Sounds is derived from Kinetics, focusing on 34 action classes
that can be recognized both visually and aurally.

Reference:
Arandjelovic & Zisserman, "Look, Listen and Learn"
ICCV 2017
"""

import os
import torch
import numpy as np
import librosa
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable, List
import torchvision.transforms as transforms


class KineticsSoundsDataset(Dataset):
    """
    Kinetics-Sounds Dataset for audio-visual action recognition.

    Expected directory structure:
    root/
        train/
            class_name/
                video_id/
                    frames/
                    audio.wav
        val/
        test/
    """

    CLASSES = [
        "blowing nose", "blowing out candles", "brushing teeth", "chopping wood",
        "coughing", "drinking", "eating chips", "eating watermelon",
        "laughing", "mowing lawn", "playing accordion", "playing bagpipes",
        "playing bass guitar", "playing drums", "playing guitar", "playing harmonica",
        "playing keyboard", "playing organ", "playing piano", "playing saxophone",
        "playing trombone", "playing trumpet", "playing violin", "playing xylophone",
        "ripping paper", "shoveling snow", "shuffling cards", "singing",
        "sneezing", "stomping grapes", "strumming guitar", "tap dancing",
        "tapping guitar", "typing"
    ]

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
            root: Root directory of Kinetics-Sounds dataset
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

        self.class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}

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
        self.samples = self._load_samples()

    def _load_samples(self) -> List:
        """Load list of samples for the split."""
        samples = []
        split_dir = os.path.join(self.root, self.split)

        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            return samples

        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            if class_name not in self.class_to_idx:
                continue

            label = self.class_to_idx[class_name]

            for video_id in os.listdir(class_path):
                video_path = os.path.join(class_path, video_id)
                if os.path.isdir(video_path):
                    samples.append((video_path, label))

        return samples

    def _load_audio(self, video_path: str) -> torch.Tensor:
        """Load and convert audio to mel spectrogram."""
        try:
            # Look for audio file
            audio_path = None
            for ext in ['.wav', '.mp3', '.flac']:
                candidate = os.path.join(video_path, f"audio{ext}")
                if os.path.exists(candidate):
                    audio_path = candidate
                    break

            if audio_path is None:
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
            print(f"Error loading audio from {video_path}: {e}")
            return torch.zeros(1, 128, 128)

    def _load_visual(self, video_path: str) -> torch.Tensor:
        """Load and stack video frames."""
        try:
            frame_dir = os.path.join(video_path, "frames")
            if not os.path.exists(frame_dir):
                frame_dir = video_path

            frames = sorted([
                f for f in os.listdir(frame_dir)
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ])

            if not frames:
                return torch.zeros(3 * self.num_frames, 224, 224)

            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)

            frame_tensors = []
            for idx in indices:
                frame_path = os.path.join(frame_dir, frames[idx])
                image = Image.open(frame_path).convert('RGB')
                frame_tensors.append(self.visual_transform(image))

            return torch.cat(frame_tensors, dim=0)

        except Exception as e:
            print(f"Error loading visual from {video_path}: {e}")
            return torch.zeros(3 * self.num_frames, 224, 224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path, label = self.samples[idx]

        audio = self._load_audio(video_path)
        visual = self._load_visual(video_path)

        return {
            "audio": audio,
            "visual": visual,
            "label": torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)
