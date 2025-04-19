# TPDL Group 19: Spoken Language Identification

## Data & Model

We utilise voice data provided by Mozilla's Common Voice corpus for this classificaion model.

### Preprocessing

Our model takes as input fixed length mel spectrograms of a vocal sample. The process to preprocess the data to this format is as follows:

 1. Convert each file from `.mp3` to `.wav`
 2. Resample audio file to 16kHz for (3)
 3. Retrieve speech segments using SpeechBrain's vad-crdnn
 4. Extract speech segments of some given minimum length
 5. Generate the mel spectrogram of the segment

###
