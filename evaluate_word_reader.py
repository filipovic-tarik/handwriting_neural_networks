import csv
import math
import os
import time

import numpy as np
from wordfreq import zipf_frequency

from nearest_neighbor_classifier import TRAINING_EXAMPLES, load_emnist_arrays
from word_reader import load_word_pixels


ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
NEIGHBORS = 5
BATCH_SIZE = 25
CHARACTER_SIZE = 28
CHARACTER_GAP = 8


# Reads image names and their correct words
def load_labeled_examples(folder):
    examples = []
    label_file = os.path.join(folder, 'labels.tsv')

    with open(label_file, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for image_name, word in reader:
            examples.append((os.path.join(folder, 'images', image_name), word))

    return examples


# Extracts the untouched 28×28 EMNIST characters from each word
def extract_test_characters(examples):
    features = []
    boundaries = []

    for number, (image_file, expected) in enumerate(examples, start=1):
        pixels = load_word_pixels(image_file)
        expected_width = (
            CHARACTER_SIZE * len(expected)
            + CHARACTER_GAP * (len(expected) - 1)
        )
        if pixels.shape != (CHARACTER_SIZE, expected_width):
            raise ValueError(f"Unexpected generated image size for {image_file}")

        start = len(features)
        for position in range(len(expected)):
            left = position * (CHARACTER_SIZE + CHARACTER_GAP)
            character = pixels[:, left:left + CHARACTER_SIZE]
            features.append(character.T.ravel() / 255)
        boundaries.append((start, len(features)))

        if number % 500 == 0 or number == len(examples):
            print(f"  prepared {number:,}/{len(examples):,} test words")

    return np.asarray(features, dtype=np.float32), boundaries


# Returns probability-like scores from the nearest EMNIST letters
def estimate_letter_probabilities(train_features, train_labels, test_features):
    probabilities = np.zeros((len(test_features), len(ALPHABET)), dtype=np.float32)
    predicted_labels = np.zeros(len(test_features), dtype=np.int16)
    train_squared = np.sum(train_features * train_features, axis=1)

    for start in range(0, len(test_features), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(test_features))
        queries = test_features[start:stop]
        query_squared = np.sum(queries * queries, axis=1)[:, None]
        distances = query_squared + train_squared[None, :] - 2 * queries @ train_features.T
        nearest = np.argpartition(distances, NEIGHBORS - 1, axis=1)[:, :NEIGHBORS]

        for offset, indices in enumerate(nearest):
            indices = indices[np.argsort(distances[offset, indices])]
            counts = {}
            for index in indices:
                label = int(train_labels[index])
                distance = max(float(distances[offset, index]), 0)
                weight = 1 / (math.sqrt(distance) + 0.001)
                counts[label] = counts.get(label, 0) + 1

                if 10 <= label <= 35:
                    probabilities[start + offset, label - 10] += weight
                elif 36 <= label <= 61:
                    probabilities[start + offset, label - 36] += weight

            predicted_labels[start + offset] = max(counts, key=counts.get)

            total = np.sum(probabilities[start + offset])
            if total > 0:
                probabilities[start + offset] /= total
            else:
                probabilities[start + offset].fill(1 / len(ALPHABET))

        if stop % 500 < BATCH_SIZE or stop == len(test_features):
            print(f"  classified {stop:,}/{len(test_features):,} test characters")

    return probabilities, predicted_labels


# Stores vocabulary words as arrays of letter numbers
def index_vocabulary_by_length(words):
    vocabulary = {}
    for word in words:
        vocabulary.setdefault(len(word), []).append(word)

    prepared = {}
    for length, candidates in vocabulary.items():
        letters = np.asarray([
            [ALPHABET.index(letter) for letter in word]
            for word in candidates
        ])
        frequencies = np.asarray([zipf_frequency(word, 'en') for word in candidates])
        prepared[length] = (candidates, letters, frequencies)
    return prepared


# Finds the vocabulary word best supported by the letter probabilities
def choose_contextual_word(probabilities, vocabulary):
    candidates, letters, frequencies = vocabulary[len(probabilities)]
    scores = np.zeros(len(candidates))

    for position in range(len(probabilities)):
        scores += np.log(probabilities[position, letters[:, position]] + 0.000001)

    scores += frequencies
    return candidates[int(np.argmax(scores))]


def report_benchmark_results(
    probabilities, predicted_labels, boundaries, examples, vocabulary
):
    raw_correct = 0
    corrected_correct = 0
    original_character_correct = 0
    character_correct = 0
    character_total = 0
    errors = []

    for (start, stop), (_, expected) in zip(boundaries, examples):
        word_probabilities = probabilities[start:stop]
        raw = ''.join(ALPHABET[index] for index in np.argmax(word_probabilities, axis=1))
        corrected = choose_contextual_word(word_probabilities, vocabulary)
        expected_labels = np.asarray([36 + ALPHABET.index(letter) for letter in expected])

        raw_correct += raw == expected
        corrected_correct += corrected == expected
        original_character_correct += np.sum(predicted_labels[start:stop] == expected_labels)
        character_correct += sum(a == b for a, b in zip(raw, expected))
        character_total += len(expected)

        if corrected != expected and len(errors) < 10:
            errors.append((expected, raw, corrected))

    print(
        f"\nOriginal 62-class character accuracy: "
        f"{100 * original_character_correct / character_total:.2f}%"
    )
    print(f"Case-insensitive letter accuracy: {100 * character_correct / character_total:.2f}%")
    print(f"Raw exact-word accuracy: {100 * raw_correct / len(examples):.2f}%")
    print(f"Vocabulary-corrected accuracy: {100 * corrected_correct / len(examples):.2f}%")

    if errors:
        print("\nExample errors:")
        for expected, raw, corrected in errors:
            print(f"  expected {expected!r}, raw {raw!r}, corrected {corrected!r}")


def run_benchmark():
    start = time.time()
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_directory, 'synthetic_words')
    training_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'test')
    emnist_file = os.path.join(base_directory, 'emnist-byclass-train.csv')

    training_examples = load_labeled_examples(training_folder)
    test_examples = load_labeled_examples(test_folder)
    words = sorted({word for _, word in training_examples})
    vocabulary = index_vocabulary_by_length(words)

    print(f"Reading all {TRAINING_EXAMPLES:,} original EMNIST training characters...")
    train_features, train_labels = load_emnist_arrays(
        emnist_file, TRAINING_EXAMPLES
    )
    print(f"Using {len(train_features):,} original EMNIST characters for KNN.")

    print(f"Preparing {len(test_examples):,} generated test words...")
    test_features, boundaries = extract_test_characters(test_examples)

    print("Running KNN letter recognition...")
    probabilities, predicted_labels = estimate_letter_probabilities(
        train_features, train_labels, test_features
    )
    report_benchmark_results(
        probabilities,
        predicted_labels,
        boundaries,
        test_examples,
        vocabulary
    )
    print(f"\nTotal time: {time.time() - start:.1f} seconds")


if __name__ == '__main__':
    run_benchmark()
