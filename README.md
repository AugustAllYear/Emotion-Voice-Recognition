# Speech-Emotion-Recognition


## Problem Statement:
Why:
Communication lost in technology. In the waves of the AI technological tsunami, many of the nuances of human communication are diminished as we translate our beings into computers. 


93% of communication in nonverbal and 38% of that is noverbal but is audatory. [Source](https://globalforum.diaglobal.org/issue/october-2018/the-power-of-nonverbal-communication-saying-everything-without-saying-anything/#:~:text=In%201971%2C%20Albert%20Mehrabian%2C%20a,how%20to%20control%20that%20impact.)


Ideally stacked with video image recognition of body language and computers could become more human communicators.


In order for future technologies such as Agentic Agents to be truly effective and possibly ease the burden for the crucial roles of nurses, caretakers, teachers ect agentic agents will need to be able to 'intuit' its audiences emotions inorder to best direct itself, call for help and know when it's up to the task at hand (not miss crucial cues!)
Thus the refinement of models that are up to the task of emotion recognition is crucial for the developing landscape of human-computer interaction.


Other applications/implications: agentic-agents, call centers, customer churn, healthcare service providers...


Goal: To accurately assess the emotional state of speakers in audio recordings.


How: Using the librosa library standardize and extract features voice signals from a dataset created for SER modeling and then build, train and test the SER model.


Deliverable: MVP - A functioning LSTM model


## Outline

Problem statement

Description

Installation

Contents

Data Sources

Code Structure

Results and Evaluation

Future Work

Acknowledgements & References

Licenses


## Description


The aim of this project is to construct and employ a Long Short Term Memory (LSTM) classification model for Speech Emotion Recognition (SER) trained on a hybrid dataset.


## Installation
Please begin by opening 01_EDA_Dataset_prep_firstModel.ipynb with [Colab](https://colab.research.google.com/)

Follow the instructions at the beginning of each notebook.
Alternatively install all libraries in the requirements.txt file in your terminal and then proceed with only the imports and file uploads as outlined in each notebook.


### Contents

```
.
├── ```
├── 01_EDA_Dataset_prep_firstModel.ipynb	
├── 02_LSTM1-4.ipynb			
├── images/
│   ├── EDA/
│   │   ├── Count of Samples per Emotion Label.png                          Example_1_Male_Actor_1_Fourier_Spectogram_Neutral.png
│   │   ├── Example_1_Male_ChromaSTFT_Emotion_ Neutral.png
│   │   ├── Example_1_Male_Mel_Spectogram_Neutral.png
│   │   ├── Example_1_Male_Spectogram_Emotion_ Neutral.png
│   │   ├── Example_2_Female_ChromaSTFT_Suprise.png
│   │   ├── Example_2_Female_Fourier_Spectogram_Suprise.png
│   │   ├── Example_2_Female_Mel_Spectogram_Suprise.png
│   │   ├── Example_2_Female_Spectogram_Suprise.png
│   │   ├── Example_3_Male_Mel_Spectogram_Sad.png
│   │   ├── Example_3_Male_Spectogram_Sad.png
│   │   ├── Example_4_Male_Mel_Spectogram_Happy.png
│   │   └── Example_4_Male_Spectogram_Happy.png
│   └── VAL/
│       ├── Confusion_Matrix_1sr_LSTM.png	
│       ├── LSTM3_Accuracy.png
│       ├── LSTM1_Accuracy.png		
│       ├── LSTM3_Loss.png
│       ├── LSTM1_Loss.png
│       ├── LSTM4_Accuracy.png			    
│       ├── LSTM2_Accuracy.png		
│       ├── LSTM4_Loss.png
│       └── LSTM2_Loss.png
├── LICENSE	  
├── models/
│   ├── lstm_model1.h5		
│   ├── lstm1_model.h5		
│   ├── lstm4_model.h5
│   ├── lstm_model2.h5		
│   ├── lstm3_model.h5		
│   └── my_lstm_model.h5
├── prepped_data/
│   ├── av-angry.m4a			
│   ├── ser-labels-paths (1).csv
│   └── mfccs3_data.npy
├── README.md
├── requirements.txt
└── Speech_emotion_recognition_presentation.pdf 
```
## Data and Sources
[Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)


[Toronto emotional speech set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)


[Surrey Audio-Visual Expressed Emotion (SAVEE)](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)


[Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)](https://www.kaggle.com/datasets/ejlok1/cremad)


[Librosa Documentation](https://librosa.org/doc/latest/index.html)


[tree.nathanfriend.com](https://tree.nathanfriend.com/?s=(%27options!(%27fancy!true~fullPath!false~trailingSlash!true~rootDot!true)~source!(%27source!%27%60%60%60701_EDA_Dataset_prep_firstModel.ipynb9702_LSTM1-4.ipynbZ97imagesZZ97LICENSE97KsZZYREADME.md7requirements.txt77images7*EDA7**Count%20of%20Samples%20per%20j%20LabelQ***********%221O_Actor_1_Fourier8U1Ozj_%20U1OwU1O8j_%20U2_FemalezJ_Fourier8JwJ8Suprise63OwSad63O8Sad64OwHappy64O8HappyQ77*VAL7**Confusion_Matrix_1sr_LSTMQ9B3GB1GZB3XB1XB4GZ9**B2GZB4XB2X7*7KsW_K1.h5ZW1kZW4kW_K2.h5ZW3kZ7*my_lstmk7Y*av-angry.m4aZ97*ser-labels-paths%20%7B1%7D.csv7*mfccs3_data.npy77%27)~version!%271%27)*%20%206Q7%227%5Cn8_Spectogram_9%5CtB7**LSTMG_AccuracyQJSuprise62_FemaleKmodelO_MaleQ.pngUNeutral6W7*lstmX_LossQY7prepped_data7Z99jEmotionk_K.h5w_Mel8z_ChromaSTFT_%22**Example_%01%22zwkjZYXWUQOKJGB9876*)


## Code Structure
The first notebook called 01_EDA_Dataset_prep_firstModel is structured in the following fashion:
Imports: import these as they are required for the notebook to run. If needed, un-comment installations and install as well. You may need to restart your notebook after installation but not imports.


Import the dataset.
Imported single files to show some different types of extractions that are possible with this type of data and necessary to transform them for modeling.
Example files illustrated the actual audio file for listening, a spectogram, a ChromaSTFT, a mel spectogram and a fourier-spectogram.


Next data labels were extracted. The labels were encoded into the file names, differently for each dataset. This required a specific function for each dataset.
At the same time we concatenated the labels and files paths, which are later used to locate the files during extraction, onto lists which would then compose our dataframe.


Next a calculation of the average, min and max length of each data file was calculated and returned to determine the best length, padding and trimming for the normalization of the data.
Later it was discovered that this was not necessary as librosa.load handles a lot of this work under the hood.


Analysis and visualization of our label classes.


Performed feature extraction of Mel-Frequency Cepstral Coefficients (MFCCs) as numerical features representing the spectral shape of sound.


Reshaped data


Encoded labels


Train test split


Built, trained and fitted on test data:
Model the LSTM: model_LSTM, 'first_lstm_model.h5 - this was the best performing model.


Stored predictions


Visualized predictions accuracy and loss against testing data


Imported a new raw audio file for classification, put through model which misclassified the audio file.


## Results and Evaluation
As can be see in Count_of_Samples_per_Emotion.png the classes are unbalanced.
It is important to note that despite stratifying in the train-test-split, the class imbalanaces negativley impacted performance.


The first model: model_LSTM, 'first_lstm_model.h5 - performed the best. It had the highest Accuracy to Validation Loss (our key metrics) ratio and also the best %'s of those metrics.
The score was 67% Accuracy and 1.22 Validation Loss (66% and 1.15 Validation Loss if the model was retrained and stopped at its best Epoch 90).
Most of the tuned models, even with early stopping added, did not improve in performance. This is clearly illustrated by the accuracy test/pred graphics as well as the loss test/pred graphic contained both in the presentation pdf and the images folder, which were created for each iteration of the model.


## Future Work
Decide whether to balance the data set by finding, imputing more values for or dropping the calm, suprised and neutral label categories.

Further tuning of the current model could help the user familiarize themself with the 
process of building an LSTM and iterating on it for performance improvements.

Both early stopping and LS regularization were implemented to improve performance but in fact both worsened the models performance this indicates it is advisable to use a simpler model.

Accessing other possible standardisations techniques, feature extraction and tuning methods would be the next step.

Utilize pretrained models such as Whisper, WavLM, and Wav2Vec 2.0, which can be fine-tuned for SER tasks. 
Any of these hould return greater results as they have been trained on large datasets and have achieved very high accuracy with SER modeling.

Transfer learning (TL) for Speaker Emotion Recognition (SER) involves taking a pre-trained machine learning model, typically trained on a related task like general audio understanding or a large-scale SER dataset, and fine-tuning it on a smaller, specific SER dataset. 


## Acknowledgments & References
Many thanks to contributors to the source material.
Many thanks to the instructors and staff at GA.


Research paper outlining the functionality, primarily signal processing and feature extraction, and use cases of the audio interpretation library Limbros: [Speech Emotion Recognition Using Librosa](https://www.aijmr.com/papers/2023/1/1003.pdf)
A wonderful article by Rohit Bohra outlining a basic potential workflow for [Emotion Detection in audio using Python — Part 1](https://medium.com/@rohitbohra23051994/emotion-detection-in-audio-using-python-6972c09054d4)
A helpful note: Review of many articles on Medium and repos on Github all of which had a unique approach and sometimes similar struggles, were very helpful in understanding workflow and potential routes. It is highly recommended to browse both of these sources when partaking in any modeling endeavor!


## Licenses
[The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio). Here is a research paper describing the data set:[The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
[Toronto emotional speech database license: Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[CREMA-D license](https://opendatacommons.org/licenses/by/1-0/index.html)
[SURREY SAVEE](https://personalpages.surrey.ac.uk/p.jackson/SAVEE/Register.html)


(did not implement but is in futurework notebook)
model1: @misc{speech-emotion-recognition,
  author = {JagjeevanAK},
  title = {Speech Emotion Recognition Model},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/JagjeevanAK/Speech-emotion-detection}
}




  
