import os.path
import numpy as np
from collections import defaultdict, OrderedDict
import torch
from neuronlp2.io.reader import CoNLLXReader
from neuronlp_kr.io.alphabet import Alphabet
from neuronlp2.io.logger import get_logger
from lib.ucorpus_parser import *


# Special vocabulary symbols - we always put them at the start.
PAD = "_PAD"
PAD_POS = "_PAD_POS"
PAD_TYPE = "_<PAD>"
PAD_MORPH = "_PAD_MORPH"
PAD_SYLL = "_PAD_SYLL"
PAD_MORPH_TAG = "_PAD_MORPH_TAG"
PAD_CHAR = "_PAD_CHAR"

ROOT = "_ROOT"
ROOT_POS = "_ROOT_POS"
ROOT_TYPE = "_<ROOT>"
ROOT_MORPH = "_ROOT_MORPH"
ROOT_SYLL = "_ROOT_SYLL"
ROOT_MORPH_TAG = "_ROOT_MORPH_TAG"
ROOT_CHAR = "_ROOT_CHAR"

END = "_END"
END_POS = "_END_POS"
END_TYPE = "_<END>"
END_MORPH = "_END_MORPH"
END_SYLL = "_END_SYLL"
END_MORPH_TAG = "_END_MORPH_TAG"
END_CHAR = "_END_CHAR"
# PAD = b"_PAD"
# PAD_POS = b"_PAD_POS"
# PAD_TYPE = b"_<PAD>"
# PAD_MORPH = b"_PAD_MORPH"
# PAD_SYLL = b"_PAD_SYLL"
# PAD_MORPH_TAG = b"_PAD_MORPH_TAG"
# PAD_CHAR = b"_PAD_CHAR"

# ROOT = b"_ROOT"
# ROOT_POS = b"_ROOT_POS"
# ROOT_TYPE = b"_<ROOT>"
# ROOT_MORPH = b"_ROOT_MORPH"
# ROOT_SYLL = b"_ROOT_SYLL"
# ROOT_MORPH_TAG = b"_ROOT_MORPH_TAG"
# ROOT_CHAR = b"_ROOT_CHAR"

# END = b"_END"
# END_POS = b"_END_POS"
# END_TYPE = b"_<END>"
# END_MORPH = b"_END_MORPH"
# END_SYLL = b"_END_SYLL"
# END_MORPH_TAG = b"_END_MORPH_TAG"
# END_CHAR = b"_END_CHAR"
_START_VOCAB = [PAD, ROOT, END]

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_MORPH = 1
PAD_ID_SYLL = 1
PAD_ID_MORPH_TAG = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 140]
_ignore_buckets = [140]




def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=100000, embedd_dict=None,
                     min_occurrence=1, normalize_digits=True):

    def expand_vocab():
        print("expand_vocab")
        # vocab_set = set(vocab_list)
        # for data_path in data_paths:
        #     # logger.info("Processing data: %s" % data_path)
        #     with open(data_path, 'r') as file:
        #         for line in file:
        #             line = line.strip()
        #             if len(line) == 0:
        #                 continue

        #             tokens = line.split('\t')
        #             for char in tokens[1]:
        #                 char_alphabet.add(char)

        #             word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
        #             pos = tokens[4]
        #             type = tokens[7]

        #             pos_alphabet.add(pos)
        #             type_alphabet.add(type)

        #             if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
        #                 vocab_set.add(word)
        #                 vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
    char_alphabet = Alphabet('character', defualt_value=True)
    pos_alphabet = Alphabet('pos')
    type_alphabet = Alphabet('type')

    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)

        char_alphabet.add(ROOT_CHAR)
        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)

        char_alphabet.add(END_CHAR)
        pos_alphabet.add(END_POS)
        type_alphabet.add(END_TYPE)

        doc, tags, dep_tuples_list, vocab = read_ucorpus_all(train_path)
 
        for sent_token in tags : 
            for word_token in sent_token :
                pos_alphabet.add(word_token[1])
        ## 추후 예정

    #     with open(train_path, 'r') as file:
    #         for line in file:
    #             line = line.strip()
    #             if len(line) == 0:
    #                 continue

    #             tokens = line.split('\t')
    #             for char in tokens[1]:
    #                 char_alphabet.add(char)

    #             word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
    #             vocab[word] += 1

    #             pos = tokens[4]
    #             pos_alphabet.add(pos)

    #             type = tokens[7]
    #             type_alphabet.add(type)



        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurrence])


        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        # if embedd_dict is not None:
            # assert isinstance(embedd_dict, OrderedDict)
            # for word in vocab.keys():
                # if word in embedd_dict or word.lower() in embedd_dict:
                    # vocab[word] += min_occurrence


        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurrence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        # if data_paths is not None and embedd_dict is not None:
        #     expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        # char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        # type_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        # char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        # type_alphabet.load(alphabet_directory)

    # print(word_alphabet.items())

    word_alphabet.close()
    # char_alphabet.close()
    pos_alphabet.close()
    # type_alphabet.close()

    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    # logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    # logger.info("Type Alphabet Size: %d" % type_alphabet.size())

    # test_print_alphabet(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet
    # return 

def test_print_alphabet(word_alphabet, char_alphabet, pos_alphabet, type_alphabet) :

    print('word instances : {}'.format(word_alphabet.instances))
    print('word instance2index : {}'.format(word_alphabet.instance2index))
    print('word items : {}'.format(word_alphabet.items()))
    print('word size : {}'.format(word_alphabet.size()))

    print('char instances : {}'.format(char_alphabet.instances))
    print('char instance2index : {}'.format(char_alphabet.instance2index))
    print('char items : {}'.format(char_alphabet.items()))
    print('char size : {}'.format(char_alphabet.size()))

    print('pos instances : {}'.format(pos_alphabet.instances))
    print('pos instance2index : {}'.format(pos_alphabet.instance2index))
    print('pos items : {}'.format(pos_alphabet.items()))
    print('pos size : {}'.format(pos_alphabet.size()))

    print('type instances : {}'.format(type_alphabet.instances))
    print('type instance2index : {}'.format(type_alphabet.instance2index))
    print('type instances items : {}'.format(type_alphabet.items()))
    print('type instances size : {}'.format(type_alphabet.size()))


