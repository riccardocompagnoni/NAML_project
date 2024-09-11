import concurrent
import os
import traceback
import pickle
import librosa
import numpy as np
import mido
import pretty_midi
from midi2audio import FluidSynth
from scipy import signal
from tqdm import tqdm
WAV_OUT = 'files/'


def construct_kernel(size, std):
    half = size // 2

    # Create a grid of (x, y) coordinates
    x = np.linspace(-half, half, size)
    y = np.linspace(-half, half, size)
    x, y = np.meshgrid(x, y)

    # Compute the Gaussian function
    gaussian = np.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))
    gaussian[:half, :half] *= -1
    gaussian[half:, half:] *= -1

    return gaussian


def novelty(roll, k_size):
    temp = np.zeros((k_size, k_size))
    novelty = np.empty(roll.shape[1] - k_size)
    kernel = construct_kernel(k_size, k_size * 30 // 64)

    for i in range(k_size):
        for j in range(i + 1, k_size):
            distance = np.linalg.norm(roll[:, i] - roll[:, j])
            temp[i, j] = distance

    novelty[0] = 2 * np.sum(temp * kernel)

    for i in range(1, roll.shape[1] - k_size):
        temp[0:k_size - 1, 0:k_size - 1] = temp[1:k_size, 1:k_size]

        for j in range(k_size - 1):
            distance = np.linalg.norm(roll[:, i + j] - roll[:, i + k_size - 1])
            temp[j, k_size - 1] = distance

        novelty[i] = 2 * np.sum(temp * kernel)

    return novelty / np.max(novelty)


def construct_similarity(curtain):
    m = curtain.shape[1]
    sim = np.empty((m, m))

    # constructing the upper triangular part, setting the diagonal to zero and copying to lower part
    for i in range(m):
        sim[i, i] = 0
        for j in range(i + 1, m):
            distance = np.linalg.norm(curtain[:, i] - curtain[:, j])
            sim[i, j] = distance
            sim[j, i] = distance
    return sim


def get_tempo_msgs(mid):
    msgs = []

    for track in mid.tracks:
        ticks = 0
        for msg in track:
            ticks += msg.time
            if msg.type == 'set_tempo':
                msgs.append([ticks, msg.tempo])
    if not msgs:
        msgs.append([0, 500000])
    msgs = np.array(msgs)
    msgs = np.sort(msgs, axis=1)
    msgs = np.vstack((msgs, [0, -1]))

    for i in range(len(msgs) - 2, 0, -1):
        msgs[i, 0] -= msgs[i - 1, 0]

    return msgs


def get_bpm(tempo_msg, ticks_per_beat, start, end):
    current_tempo = 500000
    weighted_sum = 0
    total_duration = 0
    previous_start = start

    for msg in tempo_msg:

        total_duration += mido.tick2second(msg[0], ticks_per_beat, current_tempo)

        if total_duration >= end or msg[1] == -1:
            delta_seconds = end - previous_start
            bpm = mido.tempo2bpm(current_tempo)
            weighted_sum += delta_seconds * bpm
            total_duration = end
            break

        if total_duration >= start:
            delta_seconds = total_duration - previous_start
            bpm = mido.tempo2bpm(current_tempo)
            weighted_sum += delta_seconds * bpm
            previous_start = total_duration
        current_tempo = msg[1]

    total_duration -= start

    return int(np.rint(weighted_sum / total_duration))


def remove_drum_tracks(data):

    non_drum_instruments = [
        inst for inst in data.instruments if not inst.is_drum
    ]

    new_midi = pretty_midi.PrettyMIDI()

    # Add non-drum instruments to the new MIDI object
    for inst in non_drum_instruments:
        new_midi.instruments.append(inst)

    return new_midi


def get_instruments(midi_data, start_time, end_time):
    instrument_groups = {
        'Piano': range(0, 8),
        'Chromatic Percussion': range(8, 16),
        'Organ': range(16, 24),
        'Guitar': range(24, 32),
        'Bass': range(32, 40),
        'Strings': range(40, 48),
        'Ensemble': range(48, 56),
        'Brass': range(56, 64),
        'Reed': range(64, 72),
        'Pipe': range(72, 80),
        'Synth Lead': range(80, 88),
        'Synth Pad': range(88, 96),
        'Synth Effects': range(96, 104),
        'Ethnic': range(104, 112),
        'Percussive': range(112, 120),
        'Sound Effects': range(120, 128)
    }

    instruments_playing = set()

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            if note.start < end_time and note.end > start_time:
                instruments_playing.add(instrument.program)

    # Convert instrument program numbers to instrument names
    one_hot = [0] * len(instrument_groups)

    # Iterate through the instrument groups and check if any number is in the range
    for idx, (group_name, group_range) in enumerate(instrument_groups.items()):
        # Check if any number in the list is within the current instrument group's range
        if any(num in group_range for num in instruments_playing):
            one_hot[idx] = 1

    return one_hot


def export_wav(midi_path):
    fs = FluidSynth(sound_font='resources/default.sf2')

    file_name = os.path.splitext(os.path.basename(midi_path))[0]
    curr_out = WAV_OUT + file_name + '.wav'
    fs.midi_to_audio(midi_path, curr_out)

    return curr_out


def get_mfccs(curr_out, start_time, end_time):
    y, sr = librosa.load(curr_out, offset=start_time, duration=end_time - start_time)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)

    coefficients = []
    for elem in mfccs:
        coefficients.append(np.mean(elem))
        coefficients.append(np.std(elem))

    return coefficients


def delete_wav(path):
    os.remove(path)


def produce_features(midi_path, T=10, seconds_window=20, seconds_distance=15, seconds_tolerance=5):
    feature_vector = []
    instruments = []

    # Builds piano roll from midi file and fills it with zero values both left and right to calculate novelty function
    filler = np.zeros((128, seconds_window * T // 2))
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    piano_curtain = midi_data.get_piano_roll(fs=T)
    piano_filled = np.concatenate([filler, piano_curtain, filler], axis=1)

    # Seconds_window is the time window in which novelty takes account of changes
    nov = novelty(piano_filled, seconds_window * T)

    # Novelty is padded to make sure that there is a peak both at the beginning and at the end (with tolerance seconds_distance)
    # Peaks are at least seconds_distance apart
    padded_nov = np.pad(nov, pad_width=(1, 1), mode='constant', constant_values=0)
    peaks, _ = signal.find_peaks(padded_nov, prominence=0.2, distance=seconds_distance * T)
    peaks = peaks - 1

    if peaks[0] > seconds_tolerance * T:
        peaks = np.pad(peaks, pad_width=(1, 0), mode='constant', constant_values=0)
    if peaks[-1] < len(nov) - seconds_tolerance * T:
        peaks = np.pad(peaks, pad_width=(0, 1), mode='constant', constant_values=len(nov) - 1)

    midi = mido.MidiFile(midi_path, clip=True)
    msgs = get_tempo_msgs(midi)
    ticks = midi.ticks_per_beat

    # Features are calculated on the piano roll with no drums (they have no useful or random pitch information)
    midi_filtered = remove_drum_tracks(midi_data)
    piano_filtered = midi_filtered.get_piano_roll(fs=T)

    # The midi file is converted to wav to extract mfccs coefficients. It is deleted at the end
    curr_out = export_wav(midi_path)
    for i in range(len(peaks) - 1):
        # In frames:
        start = peaks[i]
        end = peaks[i + 1]
        # In seconds:
        start_time = start / T
        end_time = end / T

        # List of all the notes played
        pitches = np.nonzero(piano_filtered[:, start:end])[0]

        # Skips the segment if it is just silence
        if len(pitches) > 0:
            avg_pitch = np.mean(pitches)
            pitch_deviation = np.std(pitches)
            max_range = np.max(pitches) - np.min(pitches)

            tempo = get_bpm(msgs, ticks, start_time, end_time)

            velocities = piano_filtered[:, start:end].flatten()
            velocities = velocities[velocities > 0.1]
            avg_intensity = np.mean(velocities)
            intensity_deviation = np.std(velocities)

            instruments.append(get_instruments(midi_data, start_time, end_time))
            duration = end_time - start_time
            mfccs = get_mfccs(curr_out, start_time, end_time)

            features = [avg_pitch, pitch_deviation, max_range, tempo, avg_intensity, intensity_deviation, duration]
            for coeff in mfccs:
                features.append(coeff)
            feature_vector.append(features)
    delete_wav(curr_out)

    return feature_vector, instruments


''' 
Extracts the features from the files in 'paths' and stores them in a list, where each element represents a song.
An element like [features, instruments] corresponds to each song
'''


def extract_features(paths, save=False, out_name='features', parallelize=False):
    results = [None] * len(paths)
    unprocessed = []

    if parallelize:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(produce_features, file): i for i, file in enumerate(paths)}

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                index = futures[future]
                try:
                    features, instruments = future.result()
                    results[index] = [features, instruments]
                except Exception as e:
                    unprocessed.append(paths[index])
                    print(f"Error processing file {index}")
                    traceback.print_exc(e)
    else:
        for index, file in enumerate(paths):
            try:
                features, instruments = produce_features(file)
                results[index] = [features, instruments]
            except Exception as e:
                unprocessed.append(paths[index])
                print(f"Error processing file {index}")
                traceback.print_exc(e)

    if unprocessed:
        print('Couldn\'t process these files:')
        for file in unprocessed:
            print(file)

    if save:
        with open(out_name + '.pkl', 'wb') as f:
            pickle.dump(results, f)

    return results


# Script used to extract the features for the training
if __name__ == '__main__':
    with open('resources/paths_train.pkl', 'rb') as f:
        paths = pickle.load(f)

    extract_features(paths, save=False, parallelize=True)
