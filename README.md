# Emotion Classification From Speech

Classification of human emotions from speech using CNN written in Python. \
Project is done during Petnica Summer Institute of Machine Learning (PSI:ML7).

DataSet used: [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) \
Model is trained on human speech spectrograms, both man and woman, which were input to 2D CNN. The network is aimed to classify the gender of the speaker and the emotion in the speech. Classes of emotions are: sad, angry, disgust, fear, surprise, calm, neutral and happy. \

Two to four layers networks where exemined and tuned based on different parametars. The best model was 4 layer CNN.

Human speech data is preprocessed using noise filters in order to get cleaner information.
