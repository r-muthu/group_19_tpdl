import os

import torch
import torchaudio
import numpy as np

from scipy.io import wavfile
import scipy.signal as sps
import soundfile

from pydub import AudioSegment
from speechbrain.inference.VAD import VAD

def reencode(file_name: str, target_name: str):
    """reencodes the given .mp3 file to .wav"""
    AudioSegment.from_mp3(file_name).export(target_name, format="wav")

def reencode_all():
    """reencodes all .mp3 files in data/mp3"""
    for lang in os.listdir("data/mp3"):
        source_dir = "data/mp3/" + lang
        target_dir = "data/wav/" + lang
        try:
            os.makedirs(target_dir)
        except FileExistsError:
            pass
        except:
            raise ValueError("file system error")

        for file in os.listdir(source_dir):
            if file.endswith(".mp3"):
                try:
                    reencode(source_dir + '/' + file, target_dir + '/' + file[:-3] + "wav")
                except:
                    print(file + " in " + source_dir + " reencoding failed!")

        print("reencoding for " + lang + " done!")
    print("reencoding done!")

def resample(file_name: str, target: int):
    """resamples the given file to the target sample rate"""
    wf, sr = torchaudio.load(file_name)
    transform = torchaudio.transforms.Resample(sr, target)
    new_wf = transform(wf)
    torchaudio.save(file_name, new_wf, target)

def resample_all(target: int):
    """resamples all .wav files in data/"""
    for lang in os.listdir("data/wav"):
        source_dir = "data/wav/" + lang

        # resample each wavfile
        for file in os.listdir(source_dir):
            if file.endswith(".wav"):
                try:
                    resample(source_dir + '/' + file, target)
                except:
                    print(file + " in " + source_dir + " resampling failed!")

        print("resampling for " + lang + " done!")
    print("resampling done!")

def generate_spectrograms(length: float):
    """generates all spectrograms from data/"""
    # load voice detection model to split audio
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

    for lang in os.listdir("data/wav"):
        source_dir = "data/wav/" + lang
        target_dir = "data/mel/" + lang
        try:
            os.makedirs(target_dir)
        except FileExistsError:
            pass
        except:
            raise ValueError("file system error")
        subseg_file_number = 0

        for file in os.listdir(source_dir):
            if file.endswith(".wav"):
                # load audio file and get segments
                file_path = source_dir + '/' + file
                wf, sr = torchaudio.load(file_path)
                sample_count = len(wf[0])
                duration = sample_count / sr
                maxseg_len = int(sr * length)
                segments = get_boundaries(vad, file_path)

                # split audio to valid sub-segments
                for segment in segments:
                    segment_len = segment[1] - segment[0]
                    segstart_idx = int(sample_count * (segment[0] / duration))
                    subseg_count = int(segment_len / length)

                    for i in range(subseg_count):
                        # extract segment and generate spectrogram
                        subseg = wf[:, segstart_idx + i * maxseg_len:segstart_idx + (i + 1) * maxseg_len]
                        subseg_spec = torchaudio.transforms.AmplitudeToDB()(to_spectrogram(subseg, sr)) # AmplitudeToDB gives log-mel spectrogram
                        save_spectrogram(target_dir + '/' + lang + str(subseg_file_number) + ".npy", subseg_spec)
                        subseg_file_number += 1

        print("spectrograms for " + lang + " done!")
    print("all spectrograms done!")

def get_boundaries(vad, file_name):
    """gets a tensor of pairs demarcating vocal segments from a model and file"""
    return vad.get_speech_segments(file_name)

def to_spectrogram(data, sr):
    """generates spectrogram from waveform slice"""
    spec_transform = torchaudio.transforms.MelSpectrogram(sr, n_mels=128)
    spec = spec_transform(data)
    return spec

def save_spectrogram(file_name, spec):
    """save spectrogram as npy image file"""
    np.save(file_name, spec.numpy())

def read_spectrogram(file_name):
    """reads a .npy spectrogram to a torch.Tensor"""
    return torch.tensor(np.load(file_name))

if __name__ == "__main__":
    # this script works; expecting all the mp3 audio input to be in `data/mp3/<lang>`
    # it will iterate through all <lang> to generate the log-mel specs for each
    reencode_all()
    resample_all(16000)
    generate_spectrograms(3)
