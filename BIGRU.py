import json
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.src.initializers import GlorotNormal
from keras.src.utils import pad_sequences
from keras.src.models import Model
from keras.src.layers import Dense, Dropout, Bidirectional, GRU, Masking, Input, Concatenate
from keras.src.utils import to_categorical
from keras.src.regularizers import L2
from keras.src.optimizers import Adam as adam
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt




def standardize_features(features, mode):

    #features[385] = np.asarray([0, 0, 0, 0, 0, 0]).reshape(1, -1)
    concatenated_features = np.vstack(features)
    means = np.mean(concatenated_features, axis=0)
    stds = np.std(concatenated_features, axis=0)

    settings = {
        'features_means': means.tolist(),
        'features_stds': stds.tolist()
    }
    with open('resources/settings_' + mode + '.json', 'w') as f:
        json.dump(settings, f)

    standardized_features = [((elem - means) / stds).tolist() for elem in features]
    return standardized_features


def merge_instruments(features, instruments):

    combined_list = [
        [features[i][j] + instruments[i][j] for j in range(len(features[i]))]
        for i in range(len(instruments))
    ]

    return combined_list


def get_features(data, mode, standardize=True):
    features = [key[0] for key in data]

    if mode == 'NO_MFCCS':

        features = [[features[i][j][:7] for j in range(len(features[i]))] for i in range(len(features))]

    elif mode == 'ORIGINAL':

        features = [[features[i][j][:6] for j in range(len(features[i]))] for i in range(len(features))]

    if standardize:
        features = standardize_features(features, mode)

    if not mode == 'ORIGINAL':
        instruments = [key[1] for key in data]
        features = merge_instruments(features, instruments)

    return features


if __name__ == '__main__':

    with open('resources/features_train.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('resources/labels_train.pkl', 'rb') as f:
        labels = pickle.load(f)

    # FULL for all features, NO_MFCCS for no mfccs, ORIGINAL for just the six described in the paper
    mode = 'FULL'
    std_features = get_features(data, mode)
    X_padded = pad_sequences(std_features, padding='post', dtype='float32')

    validation_split = 0.2
    num_classes = 5

    # Splitting the data and converting labels to one-hot
    (X_train, X_val, labels_train, labels_val) = train_test_split(
        X_padded, labels, test_size=validation_split, random_state=1312)

    y_train = to_categorical(labels_train, num_classes=num_classes)
    y_val = to_categorical(labels_val, num_classes=num_classes)

    # Model structure
    input_shape = (X_train.shape[1], X_train.shape[2])
    sequence_input = Input(shape=input_shape)

    x = Masking(mask_value=0.)(sequence_input)
    x = Bidirectional(GRU(20, return_sequences=False))(x)
    combined = Dense(13, activation='tanh', kernel_regularizer=L2(0.01), kernel_initializer=GlorotNormal())(x)
    combined = Dropout(0.3)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    # Building model
    model = Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer=adam(learning_rate=0.002), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Fitting with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.02)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=50, epochs=50, callbacks=[early_stopping])
    model.save('resources/predictor_' + mode + '.keras')

    # Model evaluation
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")

    predictions = model.predict(X_val)
    predicted_genres = np.argmax(predictions, axis=1)
    true_genres = np.argmax(y_val, axis=1)

    cm = confusion_matrix(true_genres, predicted_genres, normalize='true')

    # To normalize the matrix (by row = true genre) execute this and change normalize to true above
    cm = np.round(100*cm, 2)

    print("Confusion Matrix:")
    print(cm)

    # Plotting loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
