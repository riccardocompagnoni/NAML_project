import json
import os
import pickle
import random

'''
Takes all the files in the dataset folder, 
then builds a list of the paths to the files and a list of the corresponding labels (shuffled).
Saves in /resources
'''

if __name__ == '__main__':

    data_path = "dataset/"
    genres = ['classical', 'country', 'electronic', 'jazz', 'metal']
    # numerical label for each genre
    label = 0
    paths_dict = {}

    for genre in genres:
        folder = data_path + genre + '/'

        for file in os.listdir(folder):
            path = folder + file
            paths_dict[path] = label

        label += 1

    files = []
    labels = []

    # converting to list
    for file, genre in paths_dict.items():
        files.append(file)
        labels.append(genre)

    # if preferred can save to json
    # with open('paths_labels.json', 'w') as f:
    #     json.dump(paths_dict, f)

    # shuffling
    temp = list(zip(files, labels))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    paths, labels = list(res1), list(res2)

    with open('resources/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open('resources/paths.pkl', 'wb') as f:
        pickle.dump(paths, f)
