# Neural Network Classifier for Handwritten Numbers and Letters
Description
This project implements a simple yet effective neural network using PyTorch to classify handwritten numbers and letters into 62 different classes (10 digits + 26 uppercase letters + 26 lowercase letters). The dataset used is formatted as a CSV file, where each row represents an image with the label in the first column and 784 pixel values (28x28 images) in the subsequent columns.

Dataset
The dataset is derived from the EMNIST (Extended MNIST) dataset, specifically the "ByClass" split, which contains both digits and letters. The dataset files are:

emnist-byclass-train.csv: Training data with labels and pixel values.
emnist-byclass-test.csv: Test data with labels and pixel values.

## Word recognition

`recognize_word.py` recognizes one handwritten English word made from separated
block letters. It segments the image, classifies each letter with KNN, and uses
an English dictionary to rank the most likely words.

Run it with:

```bash
python -m pip install -r requirements.txt
python recognize_word.py word_image.png
```

The image should contain one dark handwritten word on a plain light background.
The first version requires visible gaps between letters and does not support
cursive or touching characters.
