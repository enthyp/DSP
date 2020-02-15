from collections import defaultdict
from graphviz import Graph
from uuid import uuid4

__all__ = ['symbol_freq', 'build_tree', 'build_codewords', 'get_codewords', 'dump']


class Node:
    def __init__(self, symbol, freq, left, right):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def plot(self):
        g = Graph(format='PNG')

        def _build(node, graph):
            # This assumes either both or no children!
            node_id = str(uuid4())
            if not node.left and not node.right:
                graph.node(node_id, '{}: {}'.format(node.symbol, node.freq))
                return node_id
            else:
                graph.node(node_id, '')

            l, r = _build(node.left, graph), _build(node.right, graph)
            graph.edge(node_id, l)
            graph.edge(node_id, r)

            return node_id

        _build(self, g)
        g.render('out/tree', view=True)


# For the sake of practice let's ignore heapq package :)
class Heap:
    """Min heap of key-value pairs sorted by key."""

    def __init__(self):
        self.items = []

    def insert(self, key, value):
        self.items.append((key, value))
        self.increase_key(len(self.items) - 1)

    def extract_min(self):
        if len(self.items) == 0:
            return None
        if len(self.items) == 1:
            return self.items.pop(-1)

        k, v = self.items[0]
        self.items[0] = self.items.pop(-1)
        self.heapify()

        return k, v

    def __len__(self):
        return len(self.items)

    def increase_key(self, ind):
        while ind > 0:
            p_ind = self._parent(ind)
            if self._key(p_ind) > self._key(ind):
                self._swap(ind, p_ind)
                ind = p_ind
            else:
                break

    def heapify(self):
        if len(self.items) <= 1:
            return

        i = 0
        while self._left(i) < len(self.items):
            l_ind, r_ind = self._left(i), self._right(i)
            m_ind = l_ind
            if r_ind < len(self.items) and self._key(l_ind) > self._key(r_ind):
                m_ind = r_ind

            if self._key(i) > self._key(m_ind):
                self._swap(i, m_ind)
                i = m_ind
            else:
                break

    # Abbreviations.
    def _swap(self, i, j):
        self.items[i], self.items[j] = self.items[j], self.items[i]

    def _key(self, i):
        return self.items[i][0]

    @staticmethod
    def _parent(i):
        return (i - 1) // 2

    @staticmethod
    def _left(i):
        return 2 * i + 1

    @staticmethod
    def _right(i):
        return 2 * i + 2


def symbol_freq(text):
    freq_dict = defaultdict(int)
    for c in text:
        freq_dict[c] += 1
    return freq_dict


def build_tree(freq_dict):
    freq_heap = Heap()
    for symbol, freq in freq_dict.items():
        symbol_node = Node(symbol, freq, None, None)
        freq_heap.insert(freq, symbol_node)

    while len(freq_heap) > 1:
        f1, node1 = freq_heap.extract_min()
        f2, node2 = freq_heap.extract_min()

        merge_node = Node(None, None, node1, node2)
        freq_heap.insert(f1 + f2, merge_node)
    _, root_node = freq_heap.extract_min()

    return root_node


def build_codewords(root):
    codewords = {}
    bits = ['0']

    def _build(node):
        if not node.left and not node.right:
            codewords[node.symbol] = ''.join(bits)
        else:
            bits.append('0')
            _build(node.left)

            bits[-1] = '1'
            _build(node.right)

            bits.pop(-1)

    _build(root)
    return codewords


def dump(codewords, filepath):  #
    with open(filepath, 'w') as codes_file:
        sc_pairs = [(s, c) for s, c in codewords.items()]
        sc_pairs.sort(key=lambda sc: len(sc[1]))  # by codeword length
        maps = ['{}\r{}'.format(s, c) for s, c in sc_pairs]  # just don't use CR xd
        maps_b = '\r\n'.join(maps)
        codes_file.write(maps_b)


def get_codewords(filepath):
    with open(filepath, 'r', newline='\r\n') as codes_file:
        sc_lines = codes_file.read().split('\r\n')
        sc_lines = map(lambda sc: sc.split('\r'), sc_lines)

    codewords = {k: v for k, v in sc_lines}
    return codewords
