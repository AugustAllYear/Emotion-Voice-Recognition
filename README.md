# Speech-Emotion-Recognition
Modeling Emotion Voice Recognition trained on the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)


## Problem Statement:
Why:
Communication lost in technology. In the waves of the AI technological tsunami, many of the nuances of human communication are diminished as we translate our beings into computers. 

% of communication that is noverbal but still audatory

Could be combinded with video image recognition at the same time, but lets not get ahead of ourselves!

Inorder for future techonlogies such as Agentic Agents to be truley effective and possibly supplement some of the labor intensity for jobs such as nurses, caretakers and the like these agents will need to be able to 'intuit' its audiences emotions inorder to best direct itself.  Crucial in the development of human-computer interaction.

This work cruical implications with th efollwing applications: agnetic agents, call centers, customer churn, healthcare service providers...

GOAL: To accuratley access the emotional state of the speaker of each recording file.

HOW: feature extraction of voice signals using the python library limbrosa to then classify these charateristics

DELIVERABLE:


## Outline

Problem statement
Description
Installation
Data Sources
Code Structure
Results and Evaluation
Future Work
Acknowleddgements & References
License

## Description

## Installation

requirements txt (pip freeze > requirements.txt)
Provide detailed instructions on how to set up the project on a local machine. This includes any necessary dependencies, software requirements, and installation steps. Make sure to include clear and concise instructions so that others can easily replicate your setup.

## Data sources
[Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

[Toronto emotional speech set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

[Surrey Audio-Visual Expressed Emotion (SAVEE)](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)

[Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)](https://www.kaggle.com/datasets/ejlok1/cremad)

## Code Structure
Explain the code structure and how it is organized, including any significant files and their purposes. This will help others understand how to navigate your project and find specific components.
Copy steps of notebook and expound on any confusing steps, visulisation details (how to interpret)
## Results and Evaluation
Provide an overview of the results of your project, including any relevant metrics and graphs. Include explanations of any evaluation methodologies and how they were used to assess the quality of the model. You can also make it appealing by including any pictures of your analysis or visualizations.

## Future Work
Outline potential future work that can be done to extend the project or improve its functionality. This will help others understand the scope of your project and identify areas where they can contribute.

## Acknowledgments & References
Acknowledge any contributors, data sources, or other relevant parties who have contributed to the project. This is an excellent way to show your appreciation for those who have helped you along the way.


Research paper outlining the functionality, primarily signal processing and feature extraction, and use cases of the audio interpertation library Limbros: [Speech Emotion Recognition Using Librosa](https://www.aijmr.com/papers/2023/1/1003.pdf)
A wonderful article by Rohit Bohra outlininig a basic potentail workflow for [Emotion Detection in audio using Python — Part 1](https://medium.com/@rohitbohra23051994/emotion-detection-in-audio-using-python-6972c09054d4)

## License
[The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio). Here is a research paper describing the data set:[The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
[Toronto emotional speech database license: Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[CREMA-D license](https://opendatacommons.org/licenses/by/1-0/index.html)
[SURREY SAVEE](https://personalpages.surrey.ac.uk/p.jackson/SAVEE/Register.html)



model1: @misc{speech-emotion-recognition,
  author = {JagjeevanAK},
  title = {Speech Emotion Recognition Model},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/JagjeevanAK/Speech-emotion-detection}
}
