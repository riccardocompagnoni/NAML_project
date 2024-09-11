import os
import json
import ntpath
import shutil

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)


def msd_id_to_mp3(msd_id):
    """Given an MSD ID, return the path to the corresponding mp3"""
    return os.path.join(DATA_PATH, 'msd', 'mp3',
                        msd_id_to_dirs(msd_id) + '.mp3')


def msd_id_to_h5(msd_id):
    """Given an MSD ID, return the path to the corresponding h5"""
    return os.path.join(RESULTS_PATH, 'lmd_matched_h5',
                        msd_id_to_dirs(msd_id) + '.h5')


def get_midi_path(msd_id, midi_md5, kind):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind should be one of 'matched' or 'aligned'. """
    return os.path.join(RESULTS_PATH, 'lmd_{}'.format(kind),
                        msd_id_to_dirs(msd_id), midi_md5 + '.mid')

if __name__ == '__main__':

    DATA_PATH = 'C:/Users/Riccardo/Downloads/lmd_matched'
    RESULTS_PATH = 'C:/Users/Riccardo/Downloads'
    # Path to the file match_scores.json distributed with the LMD
    SCORE_FILE = os.path.join(RESULTS_PATH, 'match_scores.json')

    with open(SCORE_FILE) as f:
        scores = json.load(f)

    labels_path = "C:/Users/Riccardo/Downloads/labels/personal/"
    dataset_path = "C:/Users/Riccardo/Downloads/dataset/"
    labels = []
    curr_genre = 0
    genres = ['electronic', 'metal', 'pop']
    random_name = 0

    for genre in genres:
        paths = []
        ids = []
        file_path = labels_path + genre + '.txt'
        end_path = dataset_path + genre

        with open(file_path) as file:
            while line := file.readline():
                md5 = line.rstrip()
                ids.append(md5)

        for msd_id in ids:
            matches = scores[msd_id]
            max_size = 0
            for midi_md5 in matches:
                # Construct the path to the aligned MIDI

                curr_path = get_midi_path(msd_id, midi_md5, 'matched')
                file_stats = os.stat(curr_path)
                curr_size = file_stats.st_size

                if curr_size > max_size:
                    max_size = curr_size
                    midi_path = curr_path

            midi_name = ntpath.basename(midi_path)
            print(midi_name)

            destination = end_path + '/' + midi_name

            if not os.path.isfile(destination):
                shutil.copy2(midi_path, destination)







