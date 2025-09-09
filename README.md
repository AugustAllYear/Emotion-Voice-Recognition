# Speech-Emotion-Recognition


## Problem Statement:
Why:
Communication lost in technology. In the waves of the AI technological tsunami, many of the nuances of human communication are diminished as we translate our beings into computers. 

93% of communication in nonverbal. 38% of that is noverbal but still audatory. [Source](https://globalforum.diaglobal.org/issue/october-2018/the-power-of-nonverbal-communication-saying-everything-without-saying-anything/#:~:text=In%201971%2C%20Albert%20Mehrabian%2C%20a,how%20to%20control%20that%20impact.)

Ideally stacked with video image recognition of body language and copmuters could become more human comminiators.

Inorder for future techonlogies such as Agentic Agents to be truley effective and possibly ease the burden for the crucial roles of nurses, caretakers, teachers ect agentic agents will need to be able to 'intuit' its audiences emotions inorder to best direct itself, call for help and know when its upto the task at hand (not miss cruical cues!)
Thus the refinement of models that are up to the task of emotion recognition is crucial for the developing landscape of human-computer interaction.

Other applications/implications: agnetic agents, call centers, customer churn, healthcare service providers...

GOAL: To accuratley access the emotional state of speakers in audio recordings.

HOW: Using the librosa library Standardize and extract features voice signals from a dataset created for SER modeling and then build, train and test the SER model.

DELIVERABLE: MVP - A functioning LSTM model

## Outline

Problem statement
Description
Installation
Data Sources
Code Structure
Results and Evaluation
Future Work
Acknowleddgements & References
Licenses

## Description

Construction and employment of an LSTM classification model for Speech Emotion Recognition trained on a hybrid of 4 datasets.

## Installation

Repo contents: 
01_EDA_Dataset_prep_firstModel.ipynb
02_LSTM1-4.ipynb


requirements txt (pip freeze > requirements.txt)
Provide detailed instructions on how to set up the project on a local machine. This includes any necessary dependencies, software requirements, and installation steps. Make sure to include clear and concise instructions so that others can easily replicate your setup.

## Data and Sources
[Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

[Toronto emotional speech set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

[Surrey Audio-Visual Expressed Emotion (SAVEE)](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)

[Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)](https://www.kaggle.com/datasets/ejlok1/cremad)

[Librosa Documentation](https://librosa.org/doc/latest/index.html)

## Code Structure
The first notebook called 01_EDA_Dataset_prep_firstModel is structured in the following fashion:
Imports: import these as the are required for the notebook to run. If needed uncode installations and install aswell. You may need to restart your notebook after installations but not imports.

Import the dataset.
Imported single files to show some different type of extractions that are possible with this type of data and necessary to transform them for modeling.
Example files illustrated the actual audio file for listening, a spectogram, a ChromaSTFT, a mel spectogram and a fourier-spectogram.

Next data labels were extracted. The labels were encoded into the fiel names, differently for each dataset. This required a specific function for each dataset.
At the same time we concatonated the labels and files paths, which are later used to locate the files during extraction, onto lists which would then compose our dataframe.

Next a calcution of the average, min and max length of each data file was calculated and returned to determine the best length, apdding and triming for the normilazation of the data.
Later it was discovered that this was not necessary as librosa.load handles alot of this work under the hood.

Analaysis and visulization of our label classes.

Performed feature extraction of Mel-Frequency Cepstral Coefficients (MFCCs) as numerical features representign the spectral shape of sound.

Model the LSTM: model_LSTM, 'first_lstm_model.h5 - this was the best performing model.

## Results and Evaluation
As can be see in the Count_of_Samples_per_Emotion.png the classes are a bit unbalanced.
It is important to note that the dataset was unbalanced, despite stratifying in the train, test, split, i beleive this impacted the performance.

The first model: model_LSTM, 'first_lstm_model.h5 - is the best performing model thus far. had the highest ratio of Accuracy to Validation Loss (our key metrics) and also the highest %'s of those metrics.
The score was 67% Accuracey and 1.22 Validation Loss (66% and 1.15 Validation Loss of we retrained it and stoped at its best Epoch 90).
Mostof the tuned models, even with early topping added did not improve in performance. This is clearly illustraed by the accuracy test/pred graphice aswell as the loss test/pred graphic contained both in the presentation pdf and the images folder, which were created for each iteration of the model.

## Future Work
Decide wheather to balance the data set by finding or imputing more values for the calm label catergory or if to drop it all together.

Further tuning of the current model could help the user familiarize themself with the 
process of building an LSTM and iterating on it for performance improvments.

Most importantly identifying why the model decresed in perfromance with most iterations seem svery important for understanding how th e model is functioning.

Asseccing other possible standardisations techniques, feature extraction and tuning methods would be the next step.

Further using a pretrained model like any listed below should return greater results as they have been trained on large datasets and have acheived very high accuracey with SER modeling.
Whisper, WavLM, and Wav2Vec 2.0, which can be fine-tuned for SER tasks. 
Transfer learning (TL) for Speaker Emotion Recognition (SER) involves taking a pre-trained machine learning model, typically trained on a related task like general audio understanding or a large-scale SER dataset, and fine-tuning it on a smaller, specific SER dataset. 

## Acknowledgments & References
Many thanks contributors to the source material.
Many thanks to the instructors and staff at GA.

Research paper outlining the functionality, primarily signal processing and feature extraction, and use cases of the audio interpertation library Limbros: [Speech Emotion Recognition Using Librosa](https://www.aijmr.com/papers/2023/1/1003.pdf)
A wonderful article by Rohit Bohra outlininig a basic potentail workflow for [Emotion Detection in audio using Python — Part 1](https://medium.com/@rohitbohra23051994/emotion-detection-in-audio-using-python-6972c09054d4)
I think it is helpful to note that I read many a article on Medium and skimmed many a repos on Github all of which had a unique approcah and sometimes similair struggles, but none the less were very helpful to see and so i would highly recommend browsing both of these sources if one hopes to partake in a similair or any modeling endevor!

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
