import torch
from torch.utils.data import Dataset
import torchvision
import torchaudio
from torchaudio import datasets, models, transforms
import torch.nn
import pandas as pd
import os
import librosa.display
import librosa.feature
import numpy as np
import scipy

import matplotlib.pyplot as plt

audio_duration = 3
sampling_rate = 22050
input_length = sampling_rate * audio_duration

label_list_women = {'neutral': 0, 'calm': 0, 'happy':1, 'sad':2, 'angry':3, 'fear':4, 'disgust':5, 'surprise':6}
label_list_men = {'neutral': 7, 'calm': 7, 'happy':8, 'sad':9, 'angry':10, 'fear':11, 'disgust':12, 'surprise':13}

class EmotionSpeechDataset(Dataset):

    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,index):

        audio_sample_path = self._get_audio_sample_path(index)

        label, gender = self._get_audio_sample_label(index)
        if gender == 'male':
            label = label_list_men[label]
        else:
            label = label_list_women[label]

        label = np.asarray(label)
        label = torch.from_numpy(label)
        label = label.long()

        signal, fs = librosa.load(audio_sample_path)
        signal, _ = librosa.effects.trim(signal, top_db = 25)

        if len(signal) > input_length:
            signal = signal[0:input_length]
        elif  input_length > len(signal):
            max_offset = input_length - len(signal)  
            signal = np.pad(signal, (0, max_offset), "constant")

        mel_spectrogram = self._return_mel_spectrogram(signal, fs)
        mel_spectrogram = torch.from_numpy(mel_spectrogram)
    
        return mel_spectrogram, label
    
    def _get_audio_sample_path(self, index):
        return self.annotations.iloc[index, 4]

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 0], self.annotations.iloc[index, 3]

    def _return_mel_spectrogram(self, signal, fs,  window_size = 20, step_size = 10):

        mel_spectrogram =  librosa.feature.melspectrogram(y = signal,
                                                          sr = fs,
                                                          n_mels = 128, n_fft=2048, hop_length=512)
        logspec = librosa.power_to_db(mel_spectrogram, ref=np.max)
        logspec = np.expand_dims(logspec, axis=0)
        return logspec
