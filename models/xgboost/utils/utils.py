import numpy as np
import torch 


__all__ = ["test_train_split", "string_pad", "shingle", "one_hot_encoder"]


def test_train_split(X, Y, frac_train, generator=None):
    assert len(X) == len(Y)
    indices = np.arange(len(X))
    if generator:
        generator.shuffle(indices)
    else:
        np.random.shuffle(indices)
    train_split_idx = int(len(X) * frac_train)
    train_idxs, test_idxs = indices[:train_split_idx], indices[train_split_idx:]

    return train_idxs, test_idxs 

def string_pad(string, total_pad_length, left_pad_length=0, pad_character="X"):
    padded_string = pad_character * left_pad_length + string
    padded_string = padded_string + pad_character * (total_pad_length - len(padded_string))

    return padded_string

def shingle(string, w):
    shingles = [string[i:i+w] for i in range(len(string) - w)]
    return shingles

def one_hot_encoder(string, value_list):
    encoded_matrix = [
            [int(letter == value) for value in value_list] for letter in string
    ]
    encoded_matrix = np.array(encoded_matrix)
    return encoded_matrix
