__author__ = 'cha'

import os.path
import random
import numpy as np
from neuronlp2.io.alphabet import Alphabet
from neuronlp2.io.logger import get_logger
from . import utils
import torch
from torch.autograd import Variable
import codecs

# Special vocabulary symbols - we always put them at the start.
PAD = b"_PAD"
PAD_POS = b"_PAD_POS"
PAD_TYPE = b"_<PAD>"
PAD_MORPH = b"_PAD_MORPH"
PAD_SYLL = b"_PAD_SYLL"
PAD_MORPH_TAG = b"_PAD_MORPH_TAG"
PAD_CHAR = b"_PAD_CHAR"

ROOT = b"_ROOT"
ROOT_POS = b"_ROOT_POS"
ROOT_TYPE = b"_<ROOT>"
ROOT_MORPH = b"_ROOT_MORPH"
ROOT_SYLL = b"_ROOT_SYLL"
ROOT_MORPH_TAG = b"_ROOT_MORPH_TAG"
ROOT_CHAR = b"_ROOT_CHAR"

END = b"_END"
END_POS = b"_END_POS"
END_TYPE = b"_<END>"
END_MORPH = b"_END_MORPH"
END_SYLL = b"_END_SYLL"
END_MORPH_TAG = b"_END_MORPH_TAG"
END_CHAR = b"_END_CHAR"
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

from .reader import SejongReader


# TODO
def extract_alphabets_conll(data_path, normalize_digits=True):
    pass

def extract_alphabets_sejong(data_path, normalize_digits=True):
    words = []
    morphs = []
    sylls = []
    morph_tags = []
    chars = []
    poses = []
    types = []

    with codecs.open(data_path, 'r', encoding='cp949') as file:
        for line in file:
            line = unicode(line)
            line = line.strip()
            if len(line) == 0:
                continue
            # add additional vocab from raw sentence
            elif line[0] == ';':
                raw_sent = line[2:]
                raw_words = utils.split_raw(raw_sent)
                raw_morph_dict = utils.get_mor_result_v2(raw_sent)

                # word + char
                if raw_morph_dict is not None:
                    raw_words += [''.join([morph.keys()[0] for morph in raw_morphs]) for raw_morphs in raw_morph_dict.values()]
                for word in raw_words:
                    word = utils.DIGIT_RE.sub(b"0", word) if normalize_digits else word
                    words.append(word)

                    for char in word:
                        chars.append(char)

                # morph + syll + morph_tag
                if raw_morph_dict is not None:
                    for raw_morphs in raw_morph_dict.values():
                        for morph_d in raw_morphs:
                            morphs_tags_list = morph_d.items()
                            morph_list = utils.get_morphs(morphs_tags_list)
                            syll_list = utils.get_sylls(morphs_tags_list)

                            # force continue when morph is empty
                            if morph_list[0] == '/':
                                continue

                            for morph in morph_list:
                                if normalize_digits:
                                    morph = utils.DIGIT_RE.sub(b"0", morph)
                                morphs.append(morph)

                            for syll in syll_list:
                                if normalize_digits:
                                    syll = utils.DIGIT_RE.sub(b"0", syll)
                                sylls.append(syll)

                            for _, morph_tag in morphs_tags_list:
                                morph_tags.append(morph_tag)

            else:
                tokens = line.split('\t')
                morphs_tags_list = utils.get_morphs_tags(tokens[4], sep='|')

                # word
                word = ''.join([morph_text for morph_text, _ in morphs_tags_list])
                word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
                words.append(word)

                # morph
                for morph in utils.get_morphs(morphs_tags_list):
                    if normalize_digits:
                        morph = utils.DIGIT_RE.sub(b"0", morph)
                    morphs.append(morph)

                # syll
                for syll in utils.get_sylls(morphs_tags_list):
                    if normalize_digits:
                        syll = utils.DIGIT_RE.sub(b"0", syll)
                    sylls.append(syll)

                # morph_tag
                for _, morph_tag in morphs_tags_list:
                    morph_tags.append(morph_tag)

                # char
                for char in word:
                    chars.append(char)

                # pos
                pos = PAD_POS
                poses.append(pos)

                # type
                types.append(tokens[2])
                
    return words, morphs, sylls, morph_tags, chars, poses, types


def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=50000, morph_embedd_dict=None, syll_embedd_dict=None,
                     min_occurence=1, normalize_digits=True, data_format='sejong'):
    def expand_vocab():
        morph_vocab_set = set(morph_vocab_list)
        syll_vocab_set = set(syll_vocab_list)
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            if data_path is None:
                continue
                
            words, morphs, sylls, morph_tags, chars, poses, types = extract_alphabets_sejong(data_path, normalize_digits=normalize_digits)
            
            for morph in morphs:
                if morph not in morph_vocab_set and (morph in morph_embedd_dict or morph.lower() in morph_embedd_dict):
                    morph_vocab_set.add(morph)
                    morph_vocab_list.append(morph)

            for syll in sylls:
                if syll not in syll_vocab_set and (syll in syll_embedd_dict or syll.lower() in syll_embedd_dict):
                    syll_vocab_set.add(syll)
                    syll_vocab_list.append(syll)
                    
            alphas_list = [words, morph_tags, chars, poses, types]
            alphabet_list = [word_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet]
            
            for alphas, alphabet in zip(alphas_list, alphabet_list):
                for alpha in alphas:
                    alphabet.add(alpha)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', defualt_value=True)
    morph_alphabet = Alphabet('morph', defualt_value=True, singleton=True)
    syll_alphabet = Alphabet('syll', defualt_value=True, singleton=True)
    morph_tag_alphabet = Alphabet('morph_tag', defualt_value=True)
    char_alphabet = Alphabet('char', defualt_value=True)
    pos_alphabet = Alphabet('pos')
    type_alphabet = Alphabet('type')
    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        word_alphabet.add(PAD)
        morph_tag_alphabet.add(PAD_MORPH_TAG)
        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)

        word_alphabet.add(ROOT)
        morph_tag_alphabet.add(ROOT_MORPH_TAG)
        char_alphabet.add(ROOT_CHAR)
        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)
        
        word_alphabet.add(END)
        morph_tag_alphabet.add(END_MORPH_TAG)
        char_alphabet.add(END_CHAR)
        pos_alphabet.add(END_POS)
        type_alphabet.add(END_TYPE)

        morph_vocab = dict()
        syll_vocab = dict()

        if data_format == 'sejong':
            words, morphs, sylls, morph_tags, chars, poses, types = \
                extract_alphabets_sejong(train_path, normalize_digits=normalize_digits)
        elif data_format == 'conll':
            words, morphs, sylls, morph_tags, chars, poses, types = \
                extract_alphabets_conll(train_path, normalize_digits=normalize_digits)

        for morph in morphs:
            morph = utils.DIGIT_RE.sub(b"0", morph) if normalize_digits else morph
            if morph in morph_vocab:
                morph_vocab[morph] += 1
            else:
                morph_vocab[morph] = 1
                
        for syll in sylls:
            syll = utils.DIGIT_RE.sub(b"0", syll) if normalize_digits else syll
            if syll in syll_vocab:
                syll_vocab[syll] += 1
            else:
                syll_vocab[syll] = 1

        alphas_list = [words, morph_tags, chars, poses, types]
        alphabet_list = [word_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet]

        for alphas, alphabet in zip(alphas_list, alphabet_list):
            for alpha in alphas:
                alphabet.add(alpha)

        # collect morph_singletons
        morph_singletons = set([morph for morph, count in morph_vocab.items() if count <= min_occurence])
        syll_singletons = set([syll for syll, count in syll_vocab.items() if count <= min_occurence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if morph_embedd_dict is not None:
            for morph in morph_vocab.keys():
                if morph in morph_embedd_dict or morph.lower() in morph_embedd_dict:
                    morph_vocab[morph] += min_occurence
                    
        if syll_embedd_dict is not None:
            for syll in syll_vocab.keys():
                if syll in syll_embedd_dict or syll.lower() in syll_embedd_dict:
                    syll_vocab[syll] += min_occurence

        morph_vocab_list = [ROOT_MORPH, PAD_MORPH, END_MORPH] + sorted(morph_vocab, key=morph_vocab.get, reverse=True)
        syll_vocab_list = [ROOT_SYLL, PAD_SYLL, END_SYLL] + sorted(syll_vocab, key=syll_vocab.get, reverse=True)
        logger.info("Total Morph Vocabulary Size: %d" % len(morph_vocab_list))
        logger.info("Total Syll Vocabulary Size: %d" % len(syll_vocab_list))
        logger.info("Total Morph Singleton Size:  %d" % len(morph_singletons))
        logger.info("Total Syll Singleton Size:  %d" % len(syll_singletons))
        morph_vocab_list = [morph for morph in morph_vocab_list if morph in [ROOT_MORPH, PAD_MORPH, END_MORPH] or morph_vocab[morph] > min_occurence]
        syll_vocab_list = [syll for syll in syll_vocab_list if syll in [ROOT_SYLL, PAD_SYLL, END_SYLL] or syll_vocab[syll] > min_occurence]
        logger.info("Total Morph Vocabulary Size (w.o rare morphs): %d" % len(morph_vocab_list))
        logger.info("Total Syll Vocabulary Size (w.o rare sylls): %d" % len(syll_vocab_list))

        if len(morph_vocab_list) > max_vocabulary_size:
            morph_vocab_list = morph_vocab_list[:max_vocabulary_size]
            
        if len(syll_vocab_list) > max_vocabulary_size:
            syll_vocab_list = syll_vocab_list[:max_vocabulary_size]

        if data_paths is not None and morph_embedd_dict is not None and syll_embedd_dict is not None:
            expand_vocab()

        for morph in morph_vocab_list:
            morph_alphabet.add(morph)
            if morph in morph_singletons:
                morph_alphabet.add_singleton(morph_alphabet.get_index(morph))
                
        for syll in syll_vocab_list:
            syll_alphabet.add(syll)
            if syll in syll_singletons:
                syll_alphabet.add_singleton(syll_alphabet.get_index(syll))

        word_alphabet.save(alphabet_directory)
        morph_alphabet.save(alphabet_directory)
        syll_alphabet.save(alphabet_directory)
        morph_tag_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        morph_alphabet.load(alphabet_directory)
        syll_alphabet.load(alphabet_directory)
        morph_tag_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)

    word_alphabet.close()
    morph_alphabet.close()
    syll_alphabet.close()
    morph_tag_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Morph Alphabet Size (Singleton): %d (%d)" % (morph_alphabet.size(), morph_alphabet.singleton_size()))
    logger.info("Syll Alphabet Size (Singleton): %d (%d)" % (syll_alphabet.size(), syll_alphabet.singleton_size()))
    logger.info("Morph tag Alphabet Size: %d" % morph_tag_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())
    return word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None,
              normalize_digits=True, symbolic_root=False, symbolic_end=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = SejongReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length


def get_batch(data, batch_size, word_alphabet=None, unk_replace=0.):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    bucket_length = _buckets[bucket_id]
    char_length = min(utils.MAX_MORPH_LENGTH, max_char_length[bucket_id] + utils.NUM_MORPH_PAD)
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([batch_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    hid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    tid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)
    single = np.zeros([batch_size, bucket_length], dtype=np.int64)

    for b in range(batch_size):
        wids, cid_seqs, pids, hids, tids = random.choice(data[bucket_id])

        inst_size = len(wids)
        # word ids
        wid_inputs[b, :inst_size] = wids
        wid_inputs[b, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[b, c, :len(cids)] = cids
            cid_inputs[b, c, len(cids):] = PAD_ID_MORPH
        cid_inputs[b, inst_size:, :] = PAD_ID_MORPH
        # pos ids
        pid_inputs[b, :inst_size] = pids
        pid_inputs[b, inst_size:] = PAD_ID_TAG
        # type ids
        tid_inputs[b, :inst_size] = tids
        tid_inputs[b, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[b, :inst_size] = hids
        hid_inputs[b, inst_size:] = PAD_ID_TAG
        # masks
        masks[b, :inst_size] = 1.0

        if unk_replace:
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[b, j] = 1

    if unk_replace:
        noise = np.random.binomial(1, unk_replace, size=[batch_size, bucket_length])
        wid_inputs = wid_inputs * (1 - noise * single)

    return wid_inputs, cid_inputs, pid_inputs, hid_inputs, tid_inputs, masks


def iterate_batch(data, batch_size, word_alphabet=None, unk_replace=0., shuffle=False):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_MORPH_LENGTH, max_char_length[bucket_id] + utils.NUM_MORPH_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_MORPH
            cid_inputs[i, inst_size:, :] = PAD_ID_MORPH
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            if unk_replace:
                for j, wid in enumerate(wids):
                    if word_alphabet.is_singleton(wid):
                        single[i, j] = 1

        if unk_replace:
            noise = np.random.binomial(1, unk_replace, size=[bucket_size, bucket_length])
            wid_inputs = wid_inputs * (1 - noise * single)

        indices = None
        if shuffle:
            indices = np.arange(bucket_size)
            np.random.shuffle(indices)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield wid_inputs[excerpt], cid_inputs[excerpt], pid_inputs[excerpt], hid_inputs[excerpt], \
                  tid_inputs[excerpt], masks[excerpt]


def read_data_to_variable(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None,
                          normalize_digits=True, symbolic_root=False, symbolic_end=False,
                          use_gpu=False, volatile=False):
    data, max_char_length = read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                      max_size=max_size, normalize_digits=normalize_digits,
                                      symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_MORPH_LENGTH, max_char_length[bucket_id] + utils.NUM_MORPH_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_MORPH
            cid_inputs[i, inst_size:, :] = PAD_ID_MORPH
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

        words = Variable(torch.from_numpy(wid_inputs), volatile=volatile)
        chars = Variable(torch.from_numpy(cid_inputs), volatile=volatile)
        pos = Variable(torch.from_numpy(pid_inputs), volatile=volatile)
        heads = Variable(torch.from_numpy(hid_inputs), volatile=volatile)
        types = Variable(torch.from_numpy(tid_inputs), volatile=volatile)
        masks = Variable(torch.from_numpy(masks), volatile=volatile)
        single = Variable(torch.from_numpy(single), volatile=volatile)
        lengths = torch.from_numpy(lengths)
        if use_gpu:
            words = words.cuda()
            chars = chars.cuda()
            pos = pos.cuda()
            heads = heads.cuda()
            types = types.cuda()
            masks = masks.cuda()
            single = single.cuda()
            lengths = lengths.cuda()

        data_variable.append((words, chars, pos, heads, types, masks, single, lengths))

    return data_variable, bucket_sizes


def get_batch_variable(data, batch_size, unk_replace=0.):
    data_variable, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    words, chars, pos, heads, types, masks, single, lengths = data_variable[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    if words.is_cuda:
        index = index.cuda()

    words = words[index]
    if unk_replace:
        ones = Variable(single.data.new(batch_size, bucket_length).fill_(1))
        noise = Variable(masks.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long())
        words = words * (ones - single[index] * noise)

    return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index]


def iterate_batch_variable(data, batch_size, unk_replace=0., shuffle=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue

        words, chars, pos, heads, types, masks, single, lengths = data_variable[bucket_id]
        if unk_replace:
            ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
            noise = Variable(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            if words.is_cuda:
                indices = indices.cuda()
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], \
                  masks[excerpt], lengths[excerpt]
