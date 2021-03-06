__author__ = 'cha'

import numpy as np
import torch
from torch.autograd import Variable
from .sejong_data import _buckets, PAD_ID_WORD, PAD_ID_MORPH, PAD_ID_TAG, UNK_ID, PAD_ID_SYLL, PAD_ID_MORPH_TAG, \
    PAD_ID_CHAR
from .sejong_data import NUM_SYMBOLIC_TAGS
from .sejong_data import create_alphabets
from neuronlp2.io.conllx_stacked_data import _generate_stack_inputs
from . import utils
from .reader import SejongReader, RawSejongReader, CoNLLXKrReader


def read_stacked_data(source_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                      max_size=None, normalize_digits=True, prior_order='deep_first', raw_reader=False, data_format='sejong'):
    data = [[] for _ in _buckets]
    max_morph_length = [0 for _ in _buckets]
    max_syll_length = [0 for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0

    if raw_reader:
        reader = RawSejongReader(source_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    elif data_format == 'sejong':
        # TODO if needed, add test reader to SejongReader
        reader = SejongReader(source_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    elif data_format == 'conll':
        reader = CoNLLXKrReader(source_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
                data[bucket_id].append(
                    [sent.word_ids, sent.morph_id_seqs, sent.syll_id_seqs, sent.morph_tag_id_seqs, sent.char_id_seqs, inst.pos_ids,
                     inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect])

                max_len = max([len(morph_seq) for morph_seq in sent.morph_seqs])
                if max_morph_length[bucket_id] < max_len:
                    max_morph_length[bucket_id] = max_len

                max_len = max([len(syll_seq) for syll_seq in sent.syll_seqs])
                if max_syll_length[bucket_id] < max_len:
                    max_syll_length[bucket_id] = max_len

                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len

                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_morph_length, max_syll_length, max_char_length


def read_stacked_data_to_variable(source_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                  max_size=None, normalize_digits=True, prior_order='deep_first', use_gpu=False, volatile=False, raw_reader=False, ignore_bucket=False, data_format='sejong'):
    # Debugging
    if ignore_bucket:
        global _buckets
        del _buckets
        from neuronlp2_kr.io.sejong_data import _ignore_buckets as _buckets

    data, max_morph_length, max_syll_length, max_char_length = \
        read_stacked_data(source_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet,
        max_size=max_size, normalize_digits=normalize_digits, prior_order=prior_order, raw_reader=raw_reader, data_format=data_format)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        morph_length = min(utils.MAX_MORPH_LENGTH, max_morph_length[bucket_id] + utils.NUM_MORPH_PAD)
        syll_length = min(utils.MAX_SYLL_LENGTH, max_syll_length[bucket_id] + utils.NUM_SYLL_PAD)
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)

        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        mid_inputs = np.empty([bucket_size, bucket_length, morph_length], dtype=np.int64) # morph
        sid_inputs = np.empty([bucket_size, bucket_length, syll_length], dtype=np.int64) # syll
        mtid_inputs = np.empty([bucket_size, bucket_length, morph_length], dtype=np.int64) # morph tag
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64) # char
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths_e = np.empty(bucket_size, dtype=np.int64)

        stack_hid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)

        masks_d = np.zeros([bucket_size, 2 * bucket_length - 1], dtype=np.float32)
        lengths_d = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, mid_seqs, sid_seqs, mtid_seqs, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst
            inst_size = len(wids)
            lengths_e[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            
            for m, mids in enumerate(mid_seqs):
                mid_inputs[i, m, :len(mids)] = mids
                mid_inputs[i, m, len(mids):] = PAD_ID_MORPH
            mid_inputs[i, inst_size:, :] = PAD_ID_MORPH
            
            for s, sids in enumerate(sid_seqs):
                sid_inputs[i, s, :len(sids)] = sids
                sid_inputs[i, s, len(sids):] = PAD_ID_SYLL
            sid_inputs[i, inst_size:, :] = PAD_ID_SYLL
            
            for mt, mtids in enumerate(mtid_seqs):
                mtid_inputs[i, mt, :len(mtids)] = mtids
                mtid_inputs[i, mt, len(mtids):] = PAD_ID_MORPH_TAG
            mtid_inputs[i, inst_size:, :] = PAD_ID_MORPH_TAG
            
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks_e
            masks_e[i, :inst_size] = 1.0

            # Debugging: disabled
            # for j, wid in enumerate(wids):
            #     if word_alphabet.is_singleton(wid):
            #         single[i, j] = 1

            inst_size_decoder = 2 * inst_size - 1
            lengths_d[i] = inst_size_decoder
            # stacked heads
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # siblings
            ssid_inputs[i, :inst_size_decoder] = ssids
            ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # skip connects
            skip_connect_inputs[i, :inst_size_decoder] = skip_ids
            skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # masks_d
            masks_d[i, :inst_size_decoder] = 1.0

        words = Variable(torch.from_numpy(wid_inputs), volatile=volatile)
        morphs = Variable(torch.from_numpy(mid_inputs), volatile=volatile)
        sylls = Variable(torch.from_numpy(sid_inputs), volatile=volatile)
        morph_tags = Variable(torch.from_numpy(mtid_inputs), volatile=volatile)
        chars = Variable(torch.from_numpy(cid_inputs), volatile=volatile)
        pos = Variable(torch.from_numpy(pid_inputs), volatile=volatile)
        heads = Variable(torch.from_numpy(hid_inputs), volatile=volatile)
        types = Variable(torch.from_numpy(tid_inputs), volatile=volatile)
        masks_e = Variable(torch.from_numpy(masks_e), volatile=volatile)
        single = Variable(torch.from_numpy(single), volatile=volatile)
        lengths_e = torch.from_numpy(lengths_e)

        stacked_heads = Variable(torch.from_numpy(stack_hid_inputs), volatile=volatile)
        children = Variable(torch.from_numpy(chid_inputs), volatile=volatile)
        siblings = Variable(torch.from_numpy(ssid_inputs), volatile=volatile)
        stacked_types = Variable(torch.from_numpy(stack_tid_inputs), volatile=volatile)
        skip_connect = torch.from_numpy(skip_connect_inputs)
        masks_d = Variable(torch.from_numpy(masks_d), volatile=volatile)
        lengths_d = torch.from_numpy(lengths_d)

        if use_gpu:
            words = words.cuda()
            morphs = morphs.cuda()
            sylls = sylls.cuda()
            morph_tags = morph_tags.cuda()
            chars = chars.cuda()
            pos = pos.cuda()
            heads = heads.cuda()
            types = types.cuda()
            masks_e = masks_e.cuda()
            single = single.cuda()
            lengths_e = lengths_e.cuda()
            stacked_heads = stacked_heads.cuda()
            children = children.cuda()
            siblings = siblings.cuda()
            stacked_types = stacked_types.cuda()
            skip_connect = skip_connect.cuda()
            masks_d = masks_d.cuda()
            lengths_d = lengths_d.cuda()

        data_variable.append(((words, morphs, sylls, morph_tags, chars, pos, heads, types, masks_e, single, lengths_e),
                              (stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d)))

    return data_variable, bucket_sizes


def get_batch_stacked_variable(data, batch_size, unk_replace=0.):
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

    data_encoder, data_decoder = data_variable[bucket_id]
    words, morphs, sylls, morph_tags, chars, pos, heads, types, masks_e, single, lengths_e = data_encoder
    stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d = data_decoder
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    if words.is_cuda:
        index = index.cuda()

    words = words[index]
    if unk_replace:
        ones = Variable(single.data.new(batch_size, bucket_length).fill_(1))
        noise = Variable(masks_e.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long())
        words = words * (ones - single[index] * noise)

    return (words, morphs[index], sylls[index], morph_tags[index], chars[index], pos[index], heads[index], types[index], masks_e[index], lengths_e[index]), \
           (stacked_heads[index], children[index], siblings[index], stacked_types[index], skip_connect[index], masks_d[index], lengths_d[index])


def iterate_batch_stacked_variable(data, batch_size, unk_replace=0., shuffle=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue
        data_encoder, data_decoder = data_variable[bucket_id]
        words, morphs, sylls, morph_tags, chars, pos, heads, types, masks_e, single, lengths_e = data_encoder
        stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d = data_decoder
        if unk_replace:
            ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
            noise = Variable(masks_e.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
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
            yield (words[excerpt], morphs[excerpt], sylls[excerpt], morph_tags[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], masks_e[excerpt], lengths_e[excerpt]), \
                  (stacked_heads[excerpt], children[excerpt], siblings[excerpt], stacked_types[excerpt], skip_connect[excerpt], masks_d[excerpt], lengths_d[excerpt])
