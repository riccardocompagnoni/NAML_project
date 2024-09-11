import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import numpy as np
from keras.src.utils import pad_sequences
from BIGRU import merge_instruments, get_features
from features import extract_features


def to_label(code):
    mapping = {
        0: 'classical',
        1: 'country',
        2: 'electronic',
        3: 'jazz',
        4: 'metal'
    }
    return mapping[code]


def normalize_features(data, settings):
    features_means = np.asarray(settings['features_means'])
    features_stds = np.asarray(settings['features_stds'])
    num_features = len(features_means)

    features = [[data[i][j][:num_features] for j in range(len(data[i]))] for i in range(len(data))]
    instruments = [[data[i][j][num_features:] for j in range(len(data[i]))] for i in range(len(data))]

    standardized_features = [((elem - features_means) / features_stds).tolist() for elem in features]

    complete_features = merge_instruments(standardized_features, instruments)

    return complete_features

'''
Runs the model obtained with the BIGRU network, according to the version in mode.
'''
if __name__ == '__main__':

    # ORIGINAL - FULL - NO_MFCCS
    mode = 'FULL'

    model = keras.models.load_model('resources/predictor_' + mode + '.keras')

    folder = 'to_predict/'
    names = os.listdir(folder)
    files = [folder + name for name in names]

    # settings file contains means and averages for the data in
    with open('resources/settings_' + mode + '.json', 'r') as f:
        settings = json.load(f)

    # Set parallelize to True for long list of files
    data = extract_features(files, parallelize=True)
    features = get_features(data, mode, standardize=False)
    std_features = normalize_features(features, settings)
    padded_features = pad_sequences(std_features, maxlen=56, padding='post', dtype='float32')

    predictions = model.predict(padded_features)
    predicted_genres = np.argmax(predictions, axis=1)

    print('\n\n')
    for i, file in enumerate(names):
        print(file, 'is predicted to be', to_label(predicted_genres[i]))


