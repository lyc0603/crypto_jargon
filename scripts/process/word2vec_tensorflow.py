"""
Script to generate word vectors from a given text file using TensorFlow.
"""

import pandas as pd
import tensorflow as tf

from environ.constants import PROCESSED_DATA_PATH


def read_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = f.readlines()
    return documents

def create_dataset(documents):
    dataset = tf.data.Dataset.from_tensor_slices(documents)
    return dataset

def preprocess_text(text):

    return text


def text_to_index(text, vocab):
    tokens = text.numpy().decode('utf-8').split()  
    return [vocab[token] if token in vocab else vocab['<UNK>'] for token in tokens]

def build_vocab(documents):
    vocab = {}
    for doc in documents:
        tokens = doc.split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    vocab['<UNK>'] = len(vocab)  
    return vocab

file_path = 'path/to/your/document_file.txt'

documents = read_document(file_path)

vocab = build_vocab(documents)

def preprocess(text):
    text = preprocess_text(text)
    return text_to_index(text, vocab)

dataset = create_dataset(documents)

dataset = dataset.map(lambda text: tf.py_function(preprocess, [text], tf.int32))

for sample in dataset.take(1):
    print(sample)
