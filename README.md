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
 

Getting the data 

 

There are two options to acquire the spectrogram data used for our model: (1) by downloading the files from Mozilla Common Voice and running the preprocessing scripts or (2) by downloading the spectrograms directly. 

To train our model with its original data, the following files need to be downloaded from https://commonvoice.mozilla.org/en/datasets: 

Language 

File 

Chinese 

Chinese (China) – Common Voice Corpus 2 

Indonesian 

Indonesian – Common Voice Corpus 5.1 

Thai 

Thai – Common Voice Corpus 6.1 

Vietnamese 

Vietnamese – Common Voice Delta Segment 10.0 

Arabic 

Arabic – Common Voice Corpus 5.1 

English 

English - Common Voice Delta Segment 15.0 

French 

French – Common Voice Delta Segment 20.0 

Hindi 

Hindi – Common Voice Delta Segment 10.0 

Spanish 

Spanish – Common Voice Corpus 2 

Table 3: Files downloaded for each language from Mozilla's Common Voice dataset. 

 

Create a location for the data files by running: 

``` 
mkdir –r data/mp3/{chinese,indo,thai,viet,arabic,english,french,hindi,spanish} 
``` 

Unzip the data into the appropriate language files within the `data/mp3` directory. For each language except English, we utilised the first 400 audio samples. For English we took the last 400 samples. To generate the log-mel-spectrograms for the model, run: 

``` 
python3 preprocessing.py 
``` 

This will take some time1. 

Alternatively, we provide a file containing copies of all the pre-processed spectrograms at https://drive.google.com/drive/folders/1kTJOxuvR3wXVKE_1nKetCILjwRyUGGZY. Create a location for the log-mel-spectrograms files by running: 

``` 
mkdir –r data/mel/{chinese,indo,thai,viet,arabic,english,french,hindi,spanish} 
``` 

Move all the data into their appropriate language files. 

 

Training the models 

 

In train.ipynb, run all cells in sequence. Do not modify cells that state ‘(no changes needed)’. First modification will occur right after the class definition of the sequence dataset, where ‘None’ needs to be replaced with the path to dataset. 

Run Trainer(model()).train() to see the output and visualisations. Each models have separate trainer initialisations. 

 

Loading the models 

Run all cells in train.ipynb that fall under the header ‘Loading and testing a model’ sequentially. Assign ‘checkpoints/ResNet50BiLSTMAttentionclass4’ to ‘model_dir’ parameter and run the cell. 
