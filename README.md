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



# Getting the Data

There are two options to acquire the spectrogram data used for our model:

1. **Download audio files** from Mozilla Common Voice and run the preprocessing scripts.
2. **Download precomputed spectrograms** directly.

---

## Option 1: Download Audio from Mozilla

To train the model with original audio data, download the following files from [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets):

| Language    | File                                               |
|-------------|----------------------------------------------------|
| Chinese     | Chinese (China) – Common Voice Corpus 2            |
| Indonesian  | Indonesian – Common Voice Corpus 5.1               |
| Thai        | Thai – Common Voice Corpus 6.1                     |
| Vietnamese  | Vietnamese – Common Voice Delta Segment 10.0       |
| Arabic      | Arabic – Common Voice Corpus 5.1                   |
| English     | English – Common Voice Delta Segment 15.0         |
| French      | French – Common Voice Delta Segment 20.0          |
| Hindi       | Hindi – Common Voice Delta Segment 10.0           |
| Spanish     | Spanish – Common Voice Corpus 2                    |

**Table 3**: Files downloaded for each language from Mozilla's Common Voice dataset.

Create folders for each language:

```bash
mkdir -p data/mp3/{chinese,indo,thai,viet,arabic,english,french,hindi,spanish}
```

Unzip the files into their respective directories under `data/mp3/`.

> For each language except English, use the **first 400 audio samples**. For English, use the **last 400 samples**.

To generate the log-mel spectrograms, run:

```bash
python3 preprocessing.py
```

> ⚠️ This will take some time.

---

## Option 2: Download Preprocessed Spectrograms

We provide preprocessed spectrograms here:  
🔗 [Google Drive - Spectrograms](https://drive.google.com/drive/folders/1kTJOxuvR3wXVKE_1nKetCILjwRyUGGZY)

Create a location for the log-mel spectrogram files:

```bash
mkdir -p data/mel/{chinese,indo,thai,viet,arabic,english,french,hindi,spanish}
```

Move each file into the appropriate language folder.

---

# Training the Models

Open `train.ipynb` and run **all cells in sequence**.

- Do **not** modify cells that state “(no changes needed)”.
- The **first required modification** is right after the class definition of the `SequenceDataset`, where `None` should be replaced with the **path to your dataset**.

## Running the Training

To train a model, run the following:

```python
Trainer(model()).train()
```

### Example

```python
model = ResNet34BiLSTMAttention(classes=num_languages)
resnet34bilstm_trainer = Trainer(
    model,
    train_loader,
    valid_loader,
    test_loader,
    num_classes=num_languages,
    num_epochs=50,
    patience=10
)
resnet34bilstm_trainer.train()
resnet34bilstm_trainer.plot_losses()
resnet34bilstm_trainer.plot_accuracies()
```

This will display the training progress along with visualizations for loss and accuracy.

> 🔁 **Note:** Each model requires its own Trainer instance and initialization.

---

## Loading and Testing the Model

To load a trained model and evaluate its performance, navigate to the **"Loading and testing a model"** section in `train.ipynb` and follow these steps:

1. Run all cells sequentially.
2. Set the `model_dir` to the folder containing the model's weights.
3. Initialize the model and call the loading/testing function.

### Example

```python
model_dir = "checkpoints/ResNet34BiLSTMAttention"
model = ResNet34BiLSTMAttention(classes=num_languages)

load_best_model_and_test(
    model_dir,
    model,
    test_loader,
    num_classes=num_languages
)
```

This will load the best checkpoint and evaluate the model on the test set.