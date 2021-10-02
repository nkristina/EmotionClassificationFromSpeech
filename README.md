# Emotion Classification From Speach

Clasification of human emotions from speech using CNN writen in Python. \
Project is done during Petnica Summer Institute of Machine Learning (PSI:ML7).

DataSet used: [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) \
Model is trained on human speech spectograms, both man and woman, which where input to 2D CNN. Network is aimed to clasifay gender of the speaker and the emotion in the speech. Classes of emotions are: sad, angry, disgust, fear, suprise, calm, neutral and happy.

Two to four layers networks where exemined and tuned based on different parametars. The best model was 4 layer CNN.

Human speech data is preprocesed using nosie filters in order to get cleaner informations.
