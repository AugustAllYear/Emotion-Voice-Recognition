# Speech-Emotion-Recognition


## Problem Statement:
Speech Emotion Recognition (SER)

In the wave of the AI technological tsunami, much of the nuance of human communication is lost as we translate our emotions and intent into data. Yet 93% of communication is nonverbal – and 38% of that is still auditory (source). When combined with video‑based body language recognition, machines could become far more human‑centric communicators.

For future technologies such as agentic agents to be truly effective – and to ease the burden on critical roles like nurses, caretakers, and teachers – these agents must be able to intuit the emotional state of the people they interact with. They need to know when to escalate, when to call for help, and when a task is beyond their current capability. Missing crucial emotional cues is not an option.

Refining models capable of robust emotion recognition is therefore essential for the evolving landscape of human‑computer interaction.

Other applications & implications

- Agentic agents & virtual assistants
- Call center analytics & customer churn prediction
- Healthcare service providers (patient monitoring, mental health)
- Education (student engagement)
- Entertainment (adaptive content)

Goal

To accurately assess the emotional state of speakers from audio recordings.

Approach

Using the librosa library, we will standardise and extract relevant vocal features from a dedicated SER dataset. We will then build, train, and test a speech emotion recognition model.

Deliverable

MVP – a functioning LSTM‑based deep learning model.

## Outline

- Problem statement
- Description
- Installation
- Contents
- Data Sources
- Code Structure
- Results and Evaluation
- Future Work
- Acknowledgements & References
- Licenses

## Description

The aim of this project is to construct and employ a Long Short‑Term Memory (LSTM) classification model for Speech Emotion Recognition (SER), trained on a hybrid dataset.

## Installation

Please begin by opening `01_EDA_Dataset_prep_firstModel.ipynb` with [Google Colab](https://colab.research.google.com/).

Follow the instructions at the beginning of each notebook.  
Alternatively, install all libraries listed in `requirements.txt` in your terminal, then proceed with only the imports and file uploads as outlined in each notebook.

## Contents

```
├── 01_EDA_Dataset_prep_firstModel.ipynb
├── 02_LSTM1-4.ipynb
├── images/
│ ├── EDA/
│ │ ├── Count of Samples per Emotion Label.png
│ │ ├── Example_1_Male_Actor_1_Fourier_Spectogram_Neutral.png
│ │ ├── Example_1_Male_ChromaSTFT_Emotion_Neutral.png
│ │ ├── Example_1_Male_Mel_Spectogram_Neutral.png
│ │ ├── Example_1_Male_Spectogram_Emotion_Neutral.png
│ │ ├── Example_2_Female_ChromaSTFT_Suprise.png
│ │ ├── Example_2_Female_Fourier_Spectogram_Suprise.png
│ │ ├── Example_2_Female_Mel_Spectogram_Suprise.png
│ │ ├── Example_2_Female_Spectogram_Suprise.png
│ │ ├── Example_3_Male_Mel_Spectogram_Sad.png
│ │ ├── Example_3_Male_Spectogram_Sad.png
│ │ ├── Example_4_Male_Mel_Spectogram_Happy.png
│ │ └── Example_4_Male_Spectogram_Happy.png
│ └── VAL/
│ ├── Confusion_Matrix_1sr_LSTM.png
│ ├── LSTM1_Accuracy.png
│ ├── LSTM1_Loss.png
│ ├── LSTM2_Accuracy.png
│ ├── LSTM2_Loss.png
│ ├── LSTM3_Accuracy.png
│ ├── LSTM3_Loss.png
│ ├── LSTM4_Accuracy.png
│ └── LSTM4_Loss.png
├── LICENSE
├── models/
│ ├── lstm_model1.h5
│ ├── lstm1_model.h5
│ ├── lstm4_model.h5
│ ├── lstm_model2.h5
│ ├── lstm3_model.h5
│ └── my_lstm_model.h5
├── prepped_data/
│ ├── av-angry.m4a
│ ├── ser-labels-paths (1).csv
│ └── mfccs3_data.npy
├── README.md
├── requirements.txt
└── Speech_emotion_recognition_presentation.pdf
```

## Data Sources

- [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- [Surrey Audio-Visual Expressed Emotion (SAVEE)](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
- [Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)](https://www.kaggle.com/datasets/ejlok1/cremad)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

## Code Structure

The first notebook, `01_EDA_Dataset_prep_firstModel.ipynb`, is structured as follows:

1. **Imports** – Import required libraries. Un‑comment installations if needed; you may need to restart the notebook after installations.

2. **Import the dataset** – Single files are loaded to demonstrate various feature extraction methods (spectrogram, ChromaSTFT, mel spectrogram, Fourier spectrogram). Example audio files can be listened to.

3. **Label extraction** – Labels are encoded in the file names, with a specific extraction function for each dataset. File paths and labels are concatenated into lists to build a DataFrame.

4. **Audio length analysis** – Calculate the average, minimum, and maximum length of each audio file to determine padding/trimming for normalisation. Later we found that `librosa.load` handles much of this automatically.

5. **Class imbalance visualisation** – Analysis and plots of label distributions.

6. **Feature extraction** – Extract Mel‑Frequency Cepstral Coefficients (MFCCs) as numerical features representing the spectral shape of sound.

7. **Data reshaping** – Reshape data to fit LSTM input requirements.

8. **Label encoding** – Encode categorical emotion labels.

9. **Train‑test split** – Stratified split to preserve class proportions.

10. **Model building & training** – Build an LSTM model (`model_LSTM`, saved as `first_lstm_model.h5`). This was the best performing model.

11. **Prediction & evaluation** – Store predictions, visualise accuracy and loss on test data.

12. **Testing on new audio** – Import a raw audio file and classify it; the model misclassified this sample.

## Results and Evaluation

As can be seen in `Count_of_Samples_per_Emotion.png`, the classes are imbalanced. Even with stratification in the train‑test split, class imbalance negatively impacted performance.

The first model (`model_LSTM`, saved as `first_lstm_model.h5`) performed best. It achieved the highest accuracy‑to‑validation‑loss ratio.  
The model reached **67% accuracy** and **1.22 validation loss** (66% accuracy and 1.15 validation loss if stopped at its best epoch 90). Most tuned models, even with early stopping, did not improve performance. This is clearly illustrated in the accuracy and loss plots (see the presentation PDF and the `images/VAL/` folder).

## Future Work

- Decide whether to balance the dataset by adding synthetic samples, removing samples, or adjusting class weights for the *calm*, *surprised*, and *neutral* categories.
- Further hyperparameter tuning could help familiarise the user with LSTM optimisation, but early stopping and L2 regularisation both worsened performance – indicating that a simpler model may be more suitable.
- Explore alternative standardisation techniques, feature extraction methods, and tuning approaches.
- Utilise pre‑trained models such as **Whisper**, **WavLM**, or **Wav2Vec 2.0**, fine‑tuned for SER. These generally achieve much higher accuracy because they are trained on large datasets.
- Apply transfer learning for SER: take a pre‑trained model (e.g., from general audio understanding or a large‑scale SER dataset) and fine‑tune it on a smaller, specific SER dataset.

## Acknowledgements & References

- Many thanks to the contributors of the source datasets and to the instructors and staff at General Assembly.
- Research paper outlining Librosa’s functionality (signal processing and feature extraction) and SER use cases: [Speech Emotion Recognition Using Librosa](https://www.aijmr.com/papers/2023/1/1003.pdf)
- A helpful article by Rohit Bohra: [Emotion Detection in Audio Using Python — Part 1](https://medium.com/@rohitbohra23051994/emotion-detection-in-audio-using-python-6972c09054d4)
- Numerous Medium articles and GitHub repositories provided valuable insight into different workflows and common challenges – highly recommended for any modelling endeavour.

## Licenses

- [The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) by Livingstone & Russo](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) is licensed under [CC BY‑NC‑SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Research paper: [The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
- [Toronto Emotional Speech Database](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) is licensed under [CC BY‑NC‑ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
- [CREMA‑D](https://www.kaggle.com/datasets/ejlok1/cremad) is licensed under the [Open Data Commons Attribution License (ODC‑By) v1.0](https://opendatacommons.org/licenses/by/1-0/).
- [SURREY SAVEE](https://personalpages.surrey.ac.uk/p.jackson/SAVEE/Register.html) – please refer to the project website for terms of use.

*Note: The pre‑trained model from Hugging Face (`JagjeevanAK/Speech-emotion-detection`) was not implemented but is referenced in the future work notebook.*
