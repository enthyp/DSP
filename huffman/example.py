from huffman.coding import Decoder, Encoder
from huffman.utils import *


def sample():
    with open('./lorem', 'r') as lorem_file:
        return lorem_file.read()


def get_tree(text):
    freq_dict = symbol_freq(text)
    return build_tree(freq_dict)


def plot():
    tree = get_tree(sample())
    tree.plot()


def encode_decode():
    text = sample()
    tree = get_tree(text)
    codewords = build_codewords(tree)
    dump(codewords, './codes')

    encoder = Encoder('./codes')
    decoder = Decoder('./codes')

    enc = encoder.encode(text)
    dec = decoder.decode(enc)
    assert dec == text  # Yay!

    comment = 'Used {} bits as opposed to {} required with ASCII encoding.'.format(
        len(enc), 8 * len(text)
    )
    print(comment)


if __name__ == '__main__':
    encode_decode()
    plot()
