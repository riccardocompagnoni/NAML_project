import numpy as np
import pretty_midi
import pickle

''' 
Checks if all the files in the paths are readable by mido and if the file contains only drums 
Saves all the corrupted files' paths, the corresponding messages and indexes.
'''

if __name__ == '__main__':

    file = []
    iteration = 0
    with open('resources/paths_train.pkl', 'rb') as f:
        paths = pickle.load(f)

    # Exceptions
    messages = []
    # Corrupted paths
    corr_paths = []
    # Indexes in the paths array
    indexes = []

    for i, midi_path in enumerate(paths):
        try:

            # Check readable
            midi_data = pretty_midi.PrettyMIDI(midi_path)

            # Check drums
            flag = True
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    flag = False
            if flag:
                raise Exception('Only drums')

            iteration += 1
            print(iteration)
        except Exception as e:
            print(repr(e))
            messages.append(repr(e))
            corr_paths.append(midi_path)
            indexes.append(i)

    print(len(corr_paths), 'Corrupted files:\n')
    print(corr_paths)

    with open('corr_msgs.pkl', 'wb') as f:
        pickle.dump(messages, f)
    with open('corr_paths.pkl', 'wb') as f:
        pickle.dump(corr_paths, f)
    with open('corr_indexes.pkl', 'wb') as f:
        pickle.dump(indexes, f)

