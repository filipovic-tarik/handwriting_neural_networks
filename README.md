# Handwritten Character and Word Recognition

## Description

This project explores four approaches to handwritten character recognition:
K-means clustering, K-nearest neighbors (KNN), a neural network implemented
with NumPy, and a neural network implemented with PyTorch. The character
classifiers distinguish 62 classes: 10 digits, 26 uppercase letters, and 26
lowercase letters. The project also includes a KNN-based recognizer for words
made from separated lowercase characters.

## Dataset

The project uses the EMNIST (Extended MNIST) ByClass split. Each CSV row
contains a label followed by 784 pixel values representing a 28×28 image. Place
these files in the repository directory before running the scripts:

- `emnist-byclass-train.csv`: training labels and pixel values
- `emnist-byclass-test.csv`: test labels and pixel values

The datasets are excluded from Git because of their size.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Word recognition

`word_reader.py` recognizes one handwritten English word made from separated
block letters. It segments the image, classifies each letter with KNN, and uses
an English dictionary to rank the most likely words.

Run it with:

```bash
python word_reader.py word_image.png
```

Generate 200,000 training images and 2,000 test images covering 50,000 words:

```bash
python build_word_dataset.py
```

Test the original EMNIST KNN on generated words with vocabulary context:

```bash
python evaluate_word_reader.py
```

The reproducible lowercase benchmark uses all 697,932 EMNIST training
characters and 2,000 word images assembled from untouched, held-out EMNIST
characters. It produced:

- 86.49% case-insensitive letter accuracy
- 44.45% exact-word accuracy from character recognition alone
- 93.90% exact-word accuracy after correction against a 50,000-word vocabulary

The vocabulary-assisted result is a closed-vocabulary measurement. The word
images contain separated lowercase characters; they do not test cursive or
touching handwriting.

The image should contain one dark handwritten word on a plain light background.
The first version requires visible gaps between letters and does not support
cursive or touching characters.
