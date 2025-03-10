import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
audio_path = "voice_10-03-2025_13-43-31.wav"  # Replace with your file
y, sr = librosa.load(audio_path, sr=None)  # Load with original sampling rate

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
output_path = "mel_spectrogram.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close()  # Close the plot to free memory

print(f"Mel spectrogram saved as {output_path}")
