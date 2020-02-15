from .utils import get_codewords


class Encoder:
    def __init__(self, file_path):
        self.codewords = get_codewords(file_path)

    def encode(self, text):
        codes = []
        for c in text:
            codes.append(self.codewords[c])
        return ''.join(codes)


class DecodingException(Exception):
    pass


class TrieNode:
    def __init__(self, val=None):
        self.val = val
        self.children = {}


class Trie:
    def __init__(self):
        self.root = TrieNode()

        # Decoding state machine.
        self.position = self.root
        self.ready = False

    def insert(self, symbol, codeword):
        def _insert(node, index):
            if index == len(codeword):
                node.val = symbol
            else:
                c = codeword[index]
                next_node = node.children.get(c, None)
                if not next_node:
                    next_node = TrieNode()
                    node.children[c] = next_node
                _insert(next_node, index + 1)

        _insert(self.root, 0)

    def feed(self, c):
        """Feed binary value to advance position."""
        if c in self.position.children:
            self.position = self.position.children[c]
            self.ready = self.position.val is not None
        else:
            raise DecodingException()

    def get(self):
        """Return a symbol if position is leaf, else None."""
        symbol = self.position.val
        if symbol:
            self.position = self.root
        return symbol


class Decoder:
    def __init__(self, file_path):
        self.codewords = get_codewords(file_path)
        self.search = Trie()
        self._build_search()

    def decode(self, text):
        symbols = []

        for c in text:
            self.search.feed(c)
            if self.search.ready:
                symbols.append(self.search.get())
        return ''.join(symbols)

    def _build_search(self):
        for symbol, codeword in self.codewords.items():
            self.search.insert(symbol, codeword)
