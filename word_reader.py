import math
import os
import sys

import numpy as np
from PIL import Image
from wordfreq import zipf_frequency

from nearest_neighbor_classifier import (
    TRAINING_EXAMPLES,
    decode_emnist_label,
    load_emnist_arrays,
)


MINIMUM_CHARACTER_GAP = 8


# Converts a word image to dark-background, light-writing format
def load_word_pixels(file_name):
    image = Image.open(file_name).convert('L')
    pixels = np.asarray(image, dtype=np.float32)

    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if np.mean(border) > 127:
        pixels = 255 - pixels

    pixels -= np.min(pixels)
    if np.max(pixels) > 0:
        pixels *= 255 / np.max(pixels)

    return pixels


# Finds a threshold that separates the writing from the background
def find_otsu_threshold(pixels):
    histogram = np.bincount(pixels.astype(np.uint8).ravel(), minlength=256)
    total = pixels.size
    total_sum = sum(i * histogram[i] for i in range(256))
    background_sum = 0
    background_count = 0
    largest_variance = -1
    threshold = 0

    for i in range(256):
        background_count += histogram[i]
        if background_count == 0:
            continue

        foreground_count = total - background_count
        if foreground_count == 0:
            break

        background_sum += i * histogram[i]
        background_mean = background_sum / background_count
        foreground_mean = (total_sum - background_sum) / foreground_count
        variance = background_count * foreground_count * (background_mean - foreground_mean) ** 2

        if variance > largest_variance:
            largest_variance = variance
            threshold = i

    return threshold


# Splits a word into characters using blank vertical gaps
def segment_characters(pixels):
    threshold = find_otsu_threshold(pixels)
    writing = pixels > threshold

    rows = np.where(np.any(writing, axis=1))[0]
    columns = np.where(np.any(writing, axis=0))[0]
    if len(rows) == 0 or len(columns) == 0:
        return []

    top = rows[0]
    bottom = rows[-1] + 1
    writing = writing[top:bottom]
    pixels = pixels[top:bottom]

    occupied_columns = np.any(writing, axis=0)
    segments = []
    start = None
    blank_start = None

    for i, occupied in enumerate(occupied_columns):
        if occupied:
            if start is None:
                start = i
            blank_start = None
        elif start is not None:
            if blank_start is None:
                blank_start = i
            if i - blank_start + 1 >= MINIMUM_CHARACTER_GAP:
                segments.append((start, blank_start))
                start = None
                blank_start = None

    if start is not None:
        segments.append((start, len(occupied_columns)))

    characters = []
    for left, right in segments:
        character_writing = writing[:, left:right]
        character_rows = np.where(np.any(character_writing, axis=1))[0]
        if len(character_rows) == 0:
            continue
        character = pixels[character_rows[0]:character_rows[-1] + 1, left:right]
        characters.append(character)

    return characters


# Centers one character in the 28×28 EMNIST format
def format_character_for_emnist(character):
    image = Image.fromarray(character.astype(np.uint8))
    width, height = image.size
    scale = min(20 / width, 20 / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    prepared = Image.new('L', (28, 28), 0)
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2
    prepared.paste(image, (left, top))

    # EMNIST stores each upright image transposed in its CSV files.
    pixels = np.asarray(prepared, dtype=np.float32).T.ravel()
    if np.max(pixels) > 0:
        pixels /= np.max(pixels)
    return pixels


# Returns several likely letters for each character image
def score_character_candidates(train_features, train_labels, characters, neighbors=50):
    queries = np.asarray([
        format_character_for_emnist(character)
        for character in characters
    ])
    train_squared = np.sum(train_features * train_features, axis=1)
    query_squared = np.sum(queries * queries, axis=1)[:, None]
    distances = query_squared + train_squared[None, :] - 2 * queries @ train_features.T
    nearest = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]

    predictions = []
    for i, indices in enumerate(nearest):
        scores = {}
        for index in indices:
            label = int(train_labels[index])
            distance = max(float(distances[i, index]), 0)
            score = 1 / (math.sqrt(distance) + 0.001)
            character = decode_emnist_label(label)
            scores[character] = scores.get(character, 0) + score

        total = sum(scores.values())
        scores = {character: score / total for character, score in scores.items()}
        predictions.append(scores)

    return predictions


# Reads an English word list
def load_dictionary_words(file_name, word_length):
    words = set()
    with open(file_name, 'r', errors='ignore') as dictionary:
        for line in dictionary:
            word = line.strip().lower()
            if len(word) == word_length and word.isalpha():
                words.add(word)
    return sorted(words)


# Scores dictionary words using the character predictions
def rank_candidate_words(predictions, words, number=5):
    letter_scores = []
    for prediction in predictions:
        scores = {}
        for character, probability in prediction.items():
            if character.isalpha():
                letter = character.lower()
                scores[letter] = scores.get(letter, 0) + probability
        letter_scores.append(scores)

    scored_words = []
    for word in words:
        score = 0
        for i, letter in enumerate(word):
            probability = letter_scores[i].get(letter, 0.000001)
            score += math.log(probability)
        score += 3 * zipf_frequency(word, 'en')
        scored_words.append((score, word))

    scored_words.sort(reverse=True)
    best_words = scored_words[:number]
    if not best_words:
        return []

    highest_score = best_words[0][0]
    weights = [math.exp(score - highest_score) for score, _ in best_words]
    total = sum(weights)
    return [(word, weight / total) for weight, (_, word) in zip(weights, best_words)]


def run_word_reader():
    if len(sys.argv) < 2:
        print("Usage: python word_reader.py word_image.png [dictionary.txt]")
        return

    base_directory = os.path.dirname(os.path.abspath(__file__))
    image_file = sys.argv[1]
    dictionary_file = sys.argv[2] if len(sys.argv) > 2 else '/usr/share/dict/words'
    training_file = os.path.join(base_directory, 'emnist-byclass-train.csv')

    print("Reading word image...")
    pixels = load_word_pixels(image_file)
    characters = segment_characters(pixels)
    if not characters:
        print("No writing was found in the image.")
        return

    print(f"Found {len(characters)} separated characters.")
    print(f"Reading all {TRAINING_EXAMPLES:,} training examples...")
    train_features, train_labels = load_emnist_arrays(training_file, TRAINING_EXAMPLES)

    print("Classifying characters...")
    predictions = score_character_candidates(train_features, train_labels, characters)
    raw_word = ''.join(max(prediction, key=prediction.get) for prediction in predictions)
    print(f"Character prediction: {raw_word}")

    words = load_dictionary_words(dictionary_file, len(characters))
    best_words = rank_candidate_words(predictions, words)
    if not best_words:
        print("No dictionary words of this length were found.")
        return

    print("\nMost likely words:")
    for word, probability in best_words:
        print(f"  {word}: {probability * 100:.1f}% relative confidence")


if __name__ == '__main__':
    run_word_reader()
