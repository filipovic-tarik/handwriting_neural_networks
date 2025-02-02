import csv
import os
import random
import sys

import numpy as np
from PIL import Image
from wordfreq import iter_wordlist

RANDOM_SEED = 20261102
NUMBER_OF_WORDS = 50000
TRAINING_VARIANTS_PER_WORD = 4
NUMBER_OF_TEST_WORDS = 2000
TRAINING_LETTERS_PER_CLASS = 1000
TEST_LETTERS_PER_CLASS = 200
MINIMUM_WORD_LENGTH = 3
MAXIMUM_WORD_LENGTH = 10
CHARACTER_GAP = 8


# Converts an EMNIST lowercase label to its letter
def decode_lowercase_label(label):
    if 36 <= label <= 61:
        return chr(ord('a') + label - 36)
    return None


# Reads a balanced selection of lowercase handwritten letters
def load_letter_image_pools(file_name, number_per_letter):
    letters = {chr(ord('a') + i): [] for i in range(26)}
    rows_read = 0

    with open(file_name, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            rows_read += 1
            label = int(row[0])
            letter = decode_lowercase_label(label)
            if letter is not None and len(letters[letter]) < number_per_letter:
                pixels = np.asarray(row[1:], dtype=np.uint8).reshape(28, 28).T
                letters[letter].append(pixels)

            if rows_read % 100000 == 0:
                smallest_pool = min(len(images) for images in letters.values())
                print(f"  read {rows_read:,} rows, at least {smallest_pool:,} examples per letter")

            if all(len(images) == number_per_letter for images in letters.values()):
                break

    for letter, images in letters.items():
        if not images:
            raise ValueError(f"No images were found for {letter!r}")

    smallest_pool = min(len(images) for images in letters.values())
    print(f"  finished with at least {smallest_pool:,} examples per letter")
    return letters


# Returns common lowercase English words
def select_common_words(number):
    words = []
    for word in iter_wordlist('en'):
        if not word.isascii() or not word.isalpha() or not word.islower():
            continue
        if not MINIMUM_WORD_LENGTH <= len(word) <= MAXIMUM_WORD_LENGTH:
            continue
        words.append(word)
        if len(words) == number:
            break
    return words


# Builds one word by joining untouched 28×28 EMNIST letters
def compose_word_image(word, letter_images, generator):
    characters = [generator.choice(letter_images[letter]) for letter in word]
    width = 28 * len(characters) + CHARACTER_GAP * (len(characters) - 1)
    canvas = Image.new('L', (width, 28), 255)
    left = 0

    for pixels in characters:
        canvas.paste(Image.fromarray(255 - pixels), (left, 0))
        left += 28 + CHARACTER_GAP

    return canvas


# Creates one word without changing any character pixels
def generate_word_image(word, letter_images, generator):
    return compose_word_image(word, letter_images, generator)


# Creates one training or test folder
def write_dataset_split(
    words, variants, held_out_words, letter_images, folder, name, generator
):
    image_folder = os.path.join(folder, 'images')
    os.makedirs(image_folder, exist_ok=True)
    label_file = os.path.join(folder, 'labels.tsv')
    number_created = 0
    expected_images = len(words) * variants - held_out_words

    with open(label_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['filename', 'text'])

        for word_number, word in enumerate(words):
            word_variants = variants - 1 if word_number < held_out_words else variants
            for variant in range(word_variants):
                file_name = f"{name}_{word_number:05d}_{variant:02d}.png"
                image = generate_word_image(word, letter_images, generator)
                image.save(os.path.join(image_folder, file_name), optimize=True)
                writer.writerow([file_name, word])
                number_created += 1

                if number_created % 5000 == 0:
                    print(f"  {name}: {number_created:,}/{expected_images:,}")


def build_dataset():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    training_file = os.path.join(base_directory, 'emnist-byclass-train.csv')
    test_file = os.path.join(base_directory, 'emnist-byclass-test.csv')
    output_folder = os.path.join(base_directory, 'synthetic_words')
    training_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    training_generator = random.Random(RANDOM_SEED)
    test_generator = random.Random(RANDOM_SEED)
    test_only = '--test-only' in sys.argv
    replace = '--replace' in sys.argv

    if os.path.exists(output_folder) and not test_only and not replace:
        raise FileExistsError("synthetic_words already exists; remove it before regenerating")

    print("Reading common English words...")
    words = select_common_words(NUMBER_OF_WORDS)
    print(f"  selected {len(words):,} words")

    if not test_only:
        print("Reading EMNIST training letters...")
        training_letters = load_letter_image_pools(
            training_file, TRAINING_LETTERS_PER_CLASS
        )
    print("Reading EMNIST test letters...")
    test_letters = load_letter_image_pools(test_file, TEST_LETTERS_PER_CLASS)

    if not test_only:
        print("Generating training words...")
        write_dataset_split(
            words,
            TRAINING_VARIANTS_PER_WORD,
            0,
            training_letters,
            training_folder,
            'train',
            training_generator
        )
    print("Generating test words...")
    write_dataset_split(
        words[:NUMBER_OF_TEST_WORDS],
        1,
        0,
        test_letters,
        test_folder,
        'test',
        test_generator
    )
    if test_only:
        print("Finished replacing the 2,000 test words with untouched EMNIST letters.")
    else:
        print("Finished generating 200,000 training and 2,000 test word images.")


if __name__ == '__main__':
    build_dataset()
