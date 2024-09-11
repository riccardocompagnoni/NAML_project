import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_features_labels(mode):
    original_list = ['avg_pitch', 'pitch_deviation', 'max_range', 'tempo',
                   'avg_intensity', 'intensity_deviation']
    if mode == 'FULL':
        return original_list + ['duration'] + [f'mfccs_coeff({i})' for i in range(10)] +[f'instrument({i})' for i in range(16)]
    else:
        return original_list

def perform_pca(processed_songs, labels, n_components=2, mode='FULL'):
    feature_labels = get_features_labels(mode)
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_songs)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # Explained variance ratio to see how much variance is captured by each principal component
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio: {explained_variance}")

    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    print(f"Cumulative Explained Variance: {cumulative_variance}")

    # Step 1: Print the linear combination of original features for each principal component, sorted by coefficient
    components = pca.components_  # Shape: (n_components, n_features)

    print("\nPrincipal Components as Sorted Linear Combinations of Original Features:")
    for i, component in enumerate(components):
        # Sort coefficients and their corresponding feature indices by absolute magnitude, in descending order
        sorted_indices = np.argsort(-np.abs(component))
        sorted_terms = [f"{component[j]:.3f} * {feature_labels[j]}" for j in sorted_indices]

        print(f"\nPC{i + 1} (sorted): {' + '.join(sorted_terms)}")

    # Step 2: Visualizing the PCA result in 2D (assuming n_components=2)
    if n_components == 2:
        plt.figure(figsize=(10, 7))
        for genre in np.unique(labels):
            plt.scatter(principal_components[labels == genre, 0],
                        principal_components[labels == genre, 1],
                        label=f"Class {genre}", alpha=0.7)

        plt.title('PCA of Processed Songs Dataset (2D)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Step 3: Visualizing in 3D (if n_components=3)
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for genre in np.unique(labels):
            ax.scatter(principal_components[labels == genre, 0],
                       principal_components[labels == genre, 1],
                       principal_components[labels == genre, 2],
                       label=f"Class {genre}", alpha=0.7)

        ax.set_title('PCA of Processed Songs Dataset (3D)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.legend()
        plt.show()

def process_song(features, instruments, mode):
    # Step 1: Compute average of the feature vectors
    features_array = np.array(features)  # Convert features to a numpy array
    avg_features = np.mean(features_array, axis=0)  # Mean of all features

    # Step 2: Sum the seventh element (index 6) instead of averaging
    avg_features[6] = np.sum(features_array[:, 6])  # Sum the seventh element

    # Step 3: Determine which instruments were present at least once
    instruments_array = np.array(instruments)  # Convert instruments to numpy array
    present_instruments = np.any(instruments_array, axis=0).astype(int)  # Logical OR across segments

    if mode == 'ORIGINAL':
        return avg_features[:6]
    else:
        # Step 4: Combine features and instruments into a single array of length 33
        combined_array = np.concatenate((avg_features, present_instruments))
        return combined_array


def process_all_songs(songs, mode):
    processed_songs = []
    for features, instruments in songs:
        processed_song = process_song(features, instruments, mode)
        processed_songs.append(processed_song)
    return processed_songs

'''
Performs PCA on the features
'''
if __name__ == '__main__':
    with open('resources/features_train.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('resources/labels_train.pkl', 'rb') as f:
        labels = pickle.load(f)

    mode = 'FULL'
    processed_songs = process_all_songs(data, mode)

    perform_pca(processed_songs, labels, n_components=2, mode=mode)
