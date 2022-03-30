import os
import json
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import glob
import torch.nn.functional as F
import random


def read_cvs(root_path='/data1/chenby/dataset/BirdClef2022', n_fold=5, k=0, phase='train'):
    def get_train_file_path(filename):
        return f"{root_path}/train_audio/{filename}"

    df = pd.read_csv(f"{root_path}/train_metadata.csv")
    df['file_path'] = df['filename'].apply(get_train_file_path)
    encoder = LabelEncoder()
    df['primary_label'] = encoder.fit_transform(df['primary_label'])

    skf = StratifiedKFold(n_splits=n_fold)
    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.primary_label)):
        df.loc[val_, "kfold"] = fold
    # 5 folds split
    print(f'total-len:{df.shape[0]}')
    if phase == 'train':
        df = df[df.kfold != k].reset_index(drop=True)
    else:
        df = df[df.kfold == k].reset_index(drop=True)
    print(f'{phase}-{k}-len:{df.shape[0]}')

    return df


class BirdCLEFDataset(Dataset):
    def __init__(self, root_path='/data1/chenby/dataset/BirdClef2022', n_fold=5, fold=0, phase='train',
                 target_sample_rate=32000, max_time=5.0, n_mels=244, n_fft=1024):
        self.phase = phase
        self.target_sample_rate = target_sample_rate
        self.num_samples = int(target_sample_rate * max_time)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.mel_spectogram = MelSpectrogram(sample_rate=self.target_sample_rate, n_mels=self.n_mels,
                                             n_fft=self.n_fft)

        df = read_cvs(root_path=root_path, n_fold=n_fold, k=fold, phase=self.phase)
        self.file_paths = df['file_path'].values
        self.labels = df['primary_label'].values

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        filepath = self.file_paths[index]
        audio, sample_rate = torchaudio.load(filepath)
        audio = self.to_mono(audio)

        if sample_rate != self.target_sample_rate:
            resample = Resample(sample_rate, self.target_sample_rate)
            audio = resample(audio)

        if audio.shape[0] > self.num_samples:
            audio = self.crop_audio(audio)

        if audio.shape[0] < self.num_samples:
            audio = self.pad_audio(audio)

        mel = self.mel_spectogram(audio)
        label = torch.tensor(self.labels[index])

        # Normalize Image
        max_val = torch.abs(mel).max()
        mel = mel / max_val
        mel = torch.unsqueeze(mel, dim=0)

        return mel, label

    def pad_audio(self, audio):
        pad_length = int(self.num_samples - audio.shape[0])
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding)
        return audio

    def crop_audio(self, audio):
        start_index = 0
        if self.phase == 'train':
            start_index = random.randint(0, audio.shape[0] - self.num_samples + 1)
        return audio[start_index: start_index + self.num_samples]

    @staticmethod
    def to_mono(audio):
        return torch.mean(audio, axis=0)

