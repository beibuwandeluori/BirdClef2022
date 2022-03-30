import warnings
warnings.filterwarnings("ignore")

import torch
import os
import numpy as np
import torchaudio
import noisereduce as nr
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader

from data.dataset import BirdCLEFDataset


def get_mel_spectrograms(audio_path, target_sample_rate=16000, reduce_noise=False, frame_length=5, device="cpu"):
    # transform = torchaudio.transforms.Spectrogram(n_fft=1800, win_length=512).to(device)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=target_sample_rate,
                                                     n_fft=1800,
                                                     hop_length=512,
                                                     n_mels=96).to(device)
    waveform, sr = torchaudio.load(audio_path)
    if reduce_noise:
        waveform = torch.tensor(nr.reduce_noise(y=waveform, sr=sr, win_length=transform.win_length, use_tqdm=True, n_jobs=2))
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, axis=0, keepdim=True)
    waveform = waveform[0]

    # Split signal into five second chunks
    spectrograms = []
    for i in range(0, len(waveform), int(frame_length * target_sample_rate)):
        split = waveform[i:i + int(frame_length * target_sample_rate)]
        if len(split) < int(frame_length * target_sample_rate):
            # num_missing_samples = int(frame_length * target_sample_rate) - split.shape[0]
            # last_dim_padding = (0, num_missing_samples)
            # split = F.pad(split, last_dim_padding)
            break

        db = torch.log(transform(split.to(device)))
        sg = np.nan_to_num(db.cpu().numpy())

        spectrograms.append(sg)

    return np.array(spectrograms)


def read_torch_audio(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    print("Shape of waveform:{}".format(waveform.size()))  # 音频大小
    print("sample rate of waveform:{}".format(sample_rate))  # 采样率
    specgram = torchaudio.transforms.MelSpectrogram()(waveform)
    print("Shape of spectrogram:{}".format(specgram.size()))


def test_data_loader(k=0, data_type='train'):
    root_path = '/data1/chenby/dataset/BirdClef2022/'
    start = time.time()
    dataset = BirdCLEFDataset(root_path=root_path, fold=k, phase=data_type)
    # dataset = AudioDatasetSubmit(root_path, load_npy=False)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
    for i, (x, y) in enumerate(train_loader):
        print(x.size(), len(y))
        if i == 10:
            break

    end = time.time()
    print('End iterate, DataLoader total time: %fs' % (end - start))


if __name__ == "__main__":
    # test_data_loader()
    test_data_loader(k=0, data_type='val')

    audio_path = './data/sampels/akekee/XC27320.ogg'
    # audio_path = './data/sampels/rettro/XC620178.ogg'

    times = 10
    start = time.time()
    for _ in range(times):
        mels = get_mel_spectrograms(audio_path=audio_path)
    print(mels.shape)
    print((time.time() - start)/times)