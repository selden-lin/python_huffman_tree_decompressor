"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num

def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])

def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}
    for item in text:
        d.setdefault(item, 0)
        d[item] += 1
    return d

def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    lst = []
    if len(freq_dict) == 1:
        for key in freq_dict:
            return HuffmanNode(left=HuffmanNode(key))

    for key in freq_dict:
        node = HuffmanNode(key)
        lst.append(node)
        node.number = freq_dict[key]

    while len(lst) != 1:
        lst = sorted(lst, key=lambda i: i.number)
        le = lst[0]
        ri = lst[1]
        new_node = HuffmanNode(left=le, right=ri)
        new_node.number = le.number + ri.number
        lst.append(new_node)
        del lst[0]
        del lst[0]
        le.number, ri.number = None, None

    lst[0].number = None
    return lst[0]

def merge(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    sum = 0
    freq = 0
    codes = get_codes(tree)
    for key in codes.keys():
        sum += len(codes[key])*freq_dict[key]
        freq += freq_dict[key]
    return sum / freq

def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    if tree.left != None and tree.right == None and tree.symbol == None:
        return {tree.left.symbol: '0'}
    if tree.left.symbol != None:
        c1 = {tree.left.symbol: '0'}
    else:
        c1 = get_codes(tree.left)
        for key in c1:
            c1[key] = '0'+ c1[key]

    if tree.right.symbol != None:
        c2 = {tree.right.symbol: '1'}
    else:
        c2 = get_codes(tree.right)
        for key in c2:
            c2[key] = '1' + c2[key]
    return merge_two_dicts(c1, c2)

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    def traverse_list(tree):
        if tree.symbol == None and tree.left != None and tree.right == None:
            return [tree]
        elif tree.symbol != None:
            return []
        elif tree.left.symbol != None and tree.right.symbol != None:
            return [tree]
        else:
            return traverse_list(tree.left) + traverse_list(tree.right) + [tree]
    l = traverse_list(tree)
    for x in range(len(l)):
        l[x].number = x

def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    left = []
    right = []
    if tree.symbol != None or tree == None:
        return bytes([])
    elif tree.left != None and tree.right == None and tree.symbol == None:
        return bytes([0, tree.left.symbol, 1, 0])
    else:
        if tree.left.symbol !=  None:
            left = [0, tree.left.symbol]
        else:
            left = [1, tree.left.number]
        if tree.right.symbol !=  None:
            right = [0, tree.right.symbol]
        else:
            right = [1, tree.right.number]
    return bytes(list(tree_to_bytes(tree.left)) + list(tree_to_bytes(tree.right)) + left + right)

def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    s = ''
    for b in text:
        s += str(codes[int(b)])
    if len(s) <= 8:
        return bytes([bits_to_byte(s)])
    lst = []
    count = 0
    while count < len(s):
        lst.append(bits_to_byte(s[count:count+8]))
        count += 8
    if count < len(s):
        lst.append(bits_to_byte(s[count:]))
    return bytes(lst)

def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])

def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")

def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) + size_to_bytes(len(text)))
    #print(list(num_nodes_to_bytes(tree)), list(tree_to_bytes(tree)))
    result += generate_compressed(text, codes)
    #print(list(l))
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression



def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))

# ====================
# Other functions

def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)), HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    if len(node_lst) == 1 and node_lst[0].r_type == 1 and node_lst[0].l_type == 0:
        return HuffmanNode(None, HuffmanNode(node_lst[0].l_data), None)
    if node_lst[root_index].l_type == 0:
        left = HuffmanNode(node_lst[root_index].l_data)
    else:
        left = generate_tree_general(node_lst, node_lst[root_index].l_data)

    if node_lst[root_index].r_type == 0:
        right = HuffmanNode(node_lst[root_index].r_data)
    else:
        right = generate_tree_general(node_lst, node_lst[root_index].r_data)
    return HuffmanNode(None, left, right)

def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)), HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    if node_lst == []:
        return None
    new_l = node_lst[:]
    split = 0
    for item in range(len(node_lst)-1, -1, -1):
        if node_lst[item].r_type == 0 and node_lst[item].l_type == 0:
            split = item
            break
    root = new_l.pop()
    if len(new_l) == 0:
        return HuffmanNode(None, HuffmanNode(root.l_data), HuffmanNode(root.r_data))

    left = new_l[:split]
    right = new_l[split:]

    if root.l_type == 0:
        l = HuffmanNode(root.l_data)
    else:
        if left == []:
            l = generate_tree_postorder(right, 2)
        else:
            l = generate_tree_postorder(left, 2)
    if root.r_type == 0:
        r = HuffmanNode(root.r_data)
    else:
        if right == []:
            r = generate_tree_postorder(left, 2)
        else:
            r = generate_tree_postorder(right, 2)


    return HuffmanNode(None, l, r)

def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    i = 1
    n = 0
    re_list = []
    bits_str = ""
    for byte in text:
        bits_str += str(byte_to_bits(byte))

    while i <= size:
        node = tree
        # trace the leaf by every bit provided
        while not node.is_leaf():
            cur_bit = bits_str[n]
            if cur_bit == '1':
                node = node.right
            else:
                node = node.left
            n += 1
        re_list.append(node.symbol)
        i += 1

    return bytes(re_list)

if __name__ == "__main__":
    #import python_ta
    #python_ta.check_all(config="huffman_pyta.txt")
    # TODO: Uncomment these when you have implemented all the functions
    #import doctest
    #doctest.testmod()
    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds.".format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds.".format(fname, time.time() - start))


