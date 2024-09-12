# Automatic music genre classification from MIDI

## Introduction: 
This project tries to replicate the approach described in [Qi He's paper](https://onlinelibrary.wiley.com/doi/10.1155/2022/9668018). 
Some additional features have been added to test the effectiveness of the original feature set.

## Repository Structure

### Prediction
Three models have been constructed:
- ORIGINAL: A model with just the six features mentioned in the main part of the paper
- NO_MFCCS: This one includes also information about the duration of the segments and the instruments playing
- FULL: A model with also the mean and the standard deviation of the first five MFCCS

To run them move the midis into the *to_predict* folder and use the *predictor.py* file, after selecting the desired model.
The *resources* and *wav_temp* folders are used to load the model and store temporarily the .wav file from which MFCCS coefficients are extracted

### Training
The models have been trained using the *BIGRU* script. The dataset is in the corresponding folder and has been loaded using the *dataset_load* and *check_readable* scripts.

## Conclusions
The suspect that the high accuracy mention'ed by the paper is overstated is confirmed by the experimental results of the three models. However, the original dataset and the original structure of the neural network is not available; hence, results can differ.