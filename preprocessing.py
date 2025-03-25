import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def voiceToSpec(wav_filename):
    # Load audio file
    y, sr = librosa.load(wav_filename, sr=None)  # Load with original sampling rate

    # Convert to Mel spectrogram
    n_mels = 128  # Number of Mel bands
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    # Convert power spectrogram to decibels
    S_db = librosa.power_to_db(S, ref=np.max)

    # Plot and save the Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.tight_layout()

    # Save the figure
    png_filename = wav_filename[:-4] #get rid of the .wav
    output_path = f"spectrogram_img/{png_filename}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()  # Close the plot to free memory

    print(f"Mel spectrogram saved as {output_path}")
