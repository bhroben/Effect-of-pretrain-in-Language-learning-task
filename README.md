# Effect-of-pretrain-in-Language-learning-task

This project was developed during the course Laboratory of Computational Physics

## 1- Preprocessing pipeline from raw audio to Mel-spectrograms
In this project we develop some methods to analyse the latent space of two models trained with mel spectrograms of audio clips in different languages. The aim is to explore the effect of different pre-training.
First, we have selected some audio datasets from Common Voice (Mozilla) and we preprocess the clips into mel-spectrograms (see preprocessing notebook).

## 2- Multivariate timeseries regression with LSTM seq2seq
Then we develop an LSTM many to many that learns the features of the mel spectrograms and, given the time step t, tries to predict t+1. Each time step corresponds to ~25ms. The model is developed in Keras. To analyse the latent space, we extract the last hidden state and the cell state from the trained LSTM and we analyse these two spaces to check the effect of different pretrainings.
Thus, first we pretrain (train for one epoch) the model with a chosen language (Spanish/Italian/Japanese) low pass filtered (<500 Hz) and then we train with the full mel spectograms of a chosen language (Spanish).
We train the model to a chosen threshold and we measure the training time; then we freeze it and we pass equal-sized samples of spectrograms in different languages and we extract hidden and cell states. 

## 3- LinearSVC
The analysis is done by implementing a linear SVC in the hidden and cell space that can linearly separate Spanish from non-Spanish. This is done by using different pre-training and comparing the accuracy of the SVC. There is indeed some degree (60%) of separation between Spanish and non-Spanish points in these latent spaces, but there is no effect distinguishable from the pre-training. Using dimensionality reduction techniques such as TSNE, we try to visualise some clusters, but further analysis is needed.

Preliminary results from training times with different pre-training languages suggest that similar languages converge to the same threshold faster than dissimilar ones, but more robust statistical analysis is needed. (see slides)

## 4- Convolutional_autoencoder_for_mel_spectrograms
Finally, we develop a convolutional autoencoder that attempts to reconstruct the MEL spectrogram given as input, but it requires more hyperparameter tuning and more computational power than we have. Nevertheless, we analyse the latent space of this model in the same way as described for the LSTM, and the linear SVM gives an accuracy of ~90%, but independent of the pre-training. We also try to analyse the latent space with TSNE, but with the same results as LSTM's.
