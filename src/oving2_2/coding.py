import numpy as np


def create_coding(feature_values):
    encodings = np.identity(len(feature_values))
    encoding_dict = {}
    for i, char in enumerate(feature_values):
        encoding_dict[char] = encodings[i]

    return encoding_dict, encodings


def encode_feature(feature, encoding_dict):
    return encoding_dict[feature]


def encode_string(string, charcodes_dict):
    encoded = np.empty((len(string), len(next(iter(charcodes_dict.values())))))
    for i, char in enumerate(string):
        encoded[i] = charcodes_dict[char]

    return encoded


if __name__ == '__main__':
    codes, _ = create_coding(' abcdefghijklmnopqrstuvwxyz')
    print(codes)
    print(encode_string("magnus", codes))
