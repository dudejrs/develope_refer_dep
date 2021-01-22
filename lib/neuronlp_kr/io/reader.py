import torch
from torch.autograd import Variable

__author__ = 'cha'

from neuronlp2.io.instance import DependencyInstance
from .instance import KrSentence
from .sejong_data import ROOT, ROOT_POS, ROOT_MORPH, ROOT_SYLL, ROOT_MORPH_TAG, ROOT_CHAR, ROOT_TYPE, PAD_POS, \
    PAD_ID_WORD, PAD_ID_MORPH, PAD_ID_SYLL, PAD_ID_MORPH_TAG, PAD_ID_CHAR, PAD_ID_TAG
from .sejong_data import END, END_POS, END_MORPH, END_SYLL, END_MORPH_TAG, END_CHAR, END_TYPE
from .sejong_data import PAD, PAD_POS, PAD_MORPH, PAD_SYLL, PAD_MORPH_TAG, PAD_CHAR, PAD_TYPE
from . import utils
from neuronlp2.io.reader import CoNLLXReader
import codecs
import numpy as np
import pickle

class SejongTestReader(object):
    def __init__(self, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet):
        self.__word_alphabet = word_alphabet
        self.__morph_alphabet = morph_alphabet
        self.__syll_alphabet = syll_alphabet
        self.__morph_tag_alphabet = morph_tag_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet

    def test_getNext(self, sentence, normalize_digits=True, symbolic_root=True, symbolic_end=False):
        sentence = sentence.strip().decode('utf-8')
        morph_dict = utils.get_mor_result_v2(sentence)

        words = []
        word_ids = []
        morph_seqs = []
        morph_id_seqs = []
        syll_seqs = []
        syll_id_seqs = []
        morph_tag_seqs = []
        morph_tag_id_seqs = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))

            morph_seqs.append([ROOT_MORPH, ])
            morph_id_seqs.append([self.__morph_alphabet.get_index(ROOT_MORPH), ])

            syll_seqs.append([ROOT_SYLL, ])
            syll_id_seqs.append([self.__syll_alphabet.get_index(ROOT_SYLL), ])

            morph_tag_seqs.append([ROOT_MORPH_TAG, ])
            morph_tag_id_seqs.append([self.__morph_tag_alphabet.get_index(ROOT_MORPH_TAG), ])

            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])

            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))

        for word, word_morph_d in morph_dict.items():
            morphs_text = []
            morphs = []
            morph_ids = []
            sylls = []
            syll_ids = []
            morph_tags = []
            morph_tag_ids = []

            # always use kmat
            for morph_d in word_morph_d:
                morphs_tags_list = morph_d.items()
                morph = utils.get_morphs(morphs_tags_list)[0]
                syll = utils.get_sylls(morphs_tags_list)

                morph_text, morph_tag = morphs_tags_list[0]
                morphs_text.append(morph_text)

                if normalize_digits:
                    morph = utils.DIGIT_RE.sub(b"0", morph)
                    syll = [utils.DIGIT_RE.sub(b"0", s) for s in syll]

                morphs.append(morph)
                morph_ids.append(self.__morph_alphabet.get_index(morph))
                sylls += syll
                syll_ids += [self.__syll_alphabet.get_index(s) for s in syll]
                morph_tags.append(morph_tag)
                morph_tag_ids.append(self.__morph_tag_alphabet.get_index(morph_tag))

            if len(morphs) > utils.MAX_MORPH_LENGTH:
                morphs = morphs[:utils.MAX_MORPH_LENGTH]
                morph_ids = morph_ids[:utils.MAX_MORPH_LENGTH]
            morph_seqs.append(morphs)
            morph_id_seqs.append(morph_ids)

            if len(sylls) > utils.MAX_SYLL_LENGTH:
                sylls = sylls[:utils.MAX_SYLL_LENGTH]
                syll_ids = syll_ids[:utils.MAX_SYLL_LENGTH]
            syll_seqs.append(sylls)
            syll_id_seqs.append(syll_ids)

            if len(morph_tags) > utils.MAX_MORPH_LENGTH:
                morph_tags = morph_tags[:utils.MAX_MORPH_LENGTH]
                morph_tag_ids = morph_tag_ids[:utils.MAX_MORPH_LENGTH]
            morph_tag_seqs.append(morph_tags)
            morph_tag_id_seqs.append(morph_tag_ids)

            word = utils.DIGIT_RE.sub(b"0", word) if normalize_digits else word
            pos = PAD_POS  # CAUTION: POS will not be used now

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            char_seqs.append([char for char in word])
            char_id_seqs.append([self.__char_alphabet.get_index(char) for char in word])

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))  # CAUTION: POS will not be used not

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))

            morph_seqs.append([END_MORPH, ])
            morph_id_seqs.append([self.__morph_alphabet.get_index(END_MORPH), ])

            syll_seqs.append([END_SYLL, ])
            syll_id_seqs.append([self.__syll_alphabet.get_index(END_SYLL), ])

            morph_tag_seqs.append([END_MORPH_TAG, ])
            morph_tag_id_seqs.append([self.__morph_tag_alphabet.get_index(END_MORPH_TAG), ])

            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])

            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))

        inst = DependencyInstance(
            KrSentence(words, word_ids, morph_seqs, morph_id_seqs, syll_seqs, syll_id_seqs, morph_tag_seqs,
                       morph_tag_id_seqs, char_seqs, char_id_seqs),
            postags, pos_ids, None, None, None)

        return inst

    def test_read_data(self, sentence):
        inst = self.test_getNext(sentence)
        sent = inst.sentence

        data = [sent.word_ids, sent.morph_id_seqs, sent.syll_id_seqs, sent.morph_tag_id_seqs, sent.char_id_seqs, inst.pos_ids]
        max_morph_length = max([len(morph_seq) for morph_seq in sent.morph_seqs])
        max_syll_length = max([len(syll_seq) for syll_seq in sent.syll_seqs])
        max_char_length = max([len(char_seq) for char_seq in sent.char_seqs])
        
        return data, max_morph_length, max_syll_length, max_char_length

    def test_data_to_input(self, sentence, volatile=False, use_gpu=False):
        data, max_morph_length, max_syll_length, max_char_length = self.test_read_data(sentence)
        inst_size = len(data[0])

        wid_inputs = np.empty([1, inst_size], dtype=np.int64)
        mid_inputs = np.empty([1, inst_size, max_morph_length], dtype=np.int64)  # morph
        sid_inputs = np.empty([1, inst_size, max_syll_length], dtype=np.int64)  # syll
        mtid_inputs = np.empty([1, inst_size, max_morph_length], dtype=np.int64)  # morph tag
        cid_inputs = np.empty([1, inst_size, max_char_length], dtype=np.int64)  # char
        pid_inputs = np.empty([1, inst_size], dtype=np.int64)

        masks_e = np.zeros([1, inst_size], dtype=np.float32)
        lengths_e = np.empty(1, dtype=np.int64)

        masks_d = np.zeros([1, 2 * inst_size - 1], dtype=np.float32)
        lengths_d = np.empty(1, dtype=np.int64)


        wids, mid_seqs, sid_seqs, mtid_seqs, cid_seqs, pids = data

        wid_inputs[0, :inst_size] = wids
        wid_inputs[0, inst_size:] = PAD_ID_WORD

        for m, mids in enumerate(mid_seqs):
            mid_inputs[0, m, :len(mids)] = mids
            mid_inputs[0, m, len(mids):] = PAD_ID_MORPH
        mid_inputs[0, inst_size:, :] = PAD_ID_MORPH

        for s, sids in enumerate(sid_seqs):
            sid_inputs[0, s, :len(sids)] = sids
            sid_inputs[0, s, len(sids):] = PAD_ID_SYLL
        sid_inputs[0, inst_size:, :] = PAD_ID_SYLL

        for mt, mtids in enumerate(mtid_seqs):
            mtid_inputs[0, mt, :len(mtids)] = mtids
            mtid_inputs[0, mt, len(mtids):] = PAD_ID_MORPH_TAG
        mtid_inputs[0, inst_size:, :] = PAD_ID_MORPH_TAG

        for c, cids in enumerate(cid_seqs):
            cid_inputs[0, c, :len(cids)] = cids
            cid_inputs[0, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[0, inst_size:, :] = PAD_ID_CHAR

        # pos ids
        pid_inputs[0, :inst_size] = pids
        pid_inputs[0, inst_size:] = PAD_ID_TAG
        # lengths_e
        lengths_e[0] = inst_size
        # masks_e
        masks_e[0, :inst_size] = 1.0


        words = Variable(torch.from_numpy(wid_inputs), volatile=volatile)
        morphs = Variable(torch.from_numpy(mid_inputs), volatile=volatile)
        sylls = Variable(torch.from_numpy(sid_inputs), volatile=volatile)
        morph_tags = Variable(torch.from_numpy(mtid_inputs), volatile=volatile)
        chars = Variable(torch.from_numpy(cid_inputs), volatile=volatile)
        pos = Variable(torch.from_numpy(pid_inputs), volatile=volatile)
        masks_e = Variable(torch.from_numpy(masks_e), volatile=volatile)
        lengths_e = torch.from_numpy(lengths_e)


        if use_gpu:
            words = words.cuda()
            morphs = morphs.cuda()
            sylls = sylls.cuda()
            morph_tags = morph_tags.cuda()
            chars = chars.cuda()
            pos = pos.cuda()
            masks_e = masks_e.cuda()
            lengths_e = lengths_e.cuda()

        return words, morphs, sylls, morph_tags, chars, pos, masks_e, lengths_e


class SejongReader(CoNLLXReader):
    def __init__(self, file_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        # CAUTION: currently using char_alphabet as morph_alphabet (morph/tag)
        super(SejongReader, self).__init__(file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        self.__source_file = codecs.open(file_path, 'r', encoding='cp949')
        self.__word_alphabet = word_alphabet
        self.__morph_alphabet = morph_alphabet
        self.__syll_alphabet = syll_alphabet
        self.__morph_tag_alphabet = morph_tag_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        line = self.__source_file.readline()
        # skip multiple blank lines and comments.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        # add raw sentence information.
        if line[0] == ';':
            raw_sent = line[2:].strip()
            raw_words = utils.split_raw(raw_sent)
            raw_morph_dict = utils.get_mor_result_v2(raw_sent)
            line = self.__source_file.readline()
        else:
            raise IOError('malformed file:', self.__source_file)

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            line = unicode(line)
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        morph_seqs = []
        morph_id_seqs = []
        syll_seqs = []
        syll_id_seqs = []
        morph_tag_seqs = []
        morph_tag_id_seqs = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            
            morph_seqs.append([ROOT_MORPH, ])
            morph_id_seqs.append([self.__morph_alphabet.get_index(ROOT_MORPH), ])

            syll_seqs.append([ROOT_SYLL, ])
            syll_id_seqs.append([self.__syll_alphabet.get_index(ROOT_SYLL), ])

            morph_tag_seqs.append([ROOT_MORPH_TAG, ])
            morph_tag_id_seqs.append([self.__morph_tag_alphabet.get_index(ROOT_MORPH_TAG), ])

            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            
            heads.append(0)

        for token_idx, tokens in enumerate(lines):
            morphs_text = []
            morphs = []
            morph_ids = []
            sylls = []
            syll_ids = []
            morph_tags = []
            morph_tag_ids = []

            # use kmat if available
            if raw_morph_dict is not None and raw_morph_dict.keys() == raw_words and len(raw_words) == len(lines):
                for morph_d in raw_morph_dict[raw_words[token_idx]]:
                    morphs_tags_list = morph_d.items()
                    morph = utils.get_morphs(morphs_tags_list)[0]
                    syll = utils.get_sylls(morphs_tags_list)
                    
                    morph_text, morph_tag = morphs_tags_list[0]
                    morphs_text.append(morph_text)
                    
                    if normalize_digits:
                        morph = utils.DIGIT_RE.sub(b"0", morph)
                        syll = [utils.DIGIT_RE.sub(b"0", s) for s in syll]
                        
                    morphs.append(morph)
                    morph_ids.append(self.__morph_alphabet.get_index(morph))
                    sylls += syll
                    syll_ids += [self.__syll_alphabet.get_index(s) for s in syll]
                    morph_tags.append(morph_tag)
                    morph_tag_ids.append(self.__morph_tag_alphabet.get_index(morph_tag))
            else:
                for morph in tokens[4].split('|'):
                    morph_text, morph_tag = utils.split_plus(morph, sep='/')
                    morphs_tags_list = [(morph_text, morph_tag)]
                    morph = utils.get_morphs(morphs_tags_list)[0]
                    syll = utils.get_sylls(morphs_tags_list)

                    morph_text, morph_tag = morphs_tags_list[0]
                    morphs_text.append(morph_text)

                    if normalize_digits:
                        morph = utils.DIGIT_RE.sub(b"0", morph)
                        syll = [utils.DIGIT_RE.sub(b"0", s) for s in syll]

                    morphs.append(morph)
                    morph_ids.append(self.__morph_alphabet.get_index(morph))
                    sylls += syll
                    syll_ids += [self.__syll_alphabet.get_index(s) for s in syll]
                    morph_tags.append(morph_tag)
                    morph_tag_ids.append(self.__morph_tag_alphabet.get_index(morph_tag))

            if len(morphs) > utils.MAX_MORPH_LENGTH:
                morphs = morphs[:utils.MAX_MORPH_LENGTH]
                morph_ids = morph_ids[:utils.MAX_MORPH_LENGTH]
            morph_seqs.append(morphs)
            morph_id_seqs.append(morph_ids)
            
            if len(sylls) > utils.MAX_SYLL_LENGTH:
                sylls = sylls[:utils.MAX_SYLL_LENGTH]
                syll_ids = syll_ids[:utils.MAX_SYLL_LENGTH]
            syll_seqs.append(sylls)
            syll_id_seqs.append(syll_ids)

            if len(morph_tags) > utils.MAX_MORPH_LENGTH:
                morph_tags = morph_tags[:utils.MAX_MORPH_LENGTH]
                morph_tag_ids = morph_tag_ids[:utils.MAX_MORPH_LENGTH]
            morph_tag_seqs.append(morph_tags)
            morph_tag_id_seqs.append(morph_tag_ids)

            if len(raw_words) == len(lines):
                word = raw_words[token_idx]
            else:
                word = ''.join(morphs_text)

            word = utils.DIGIT_RE.sub(b"0", word) if normalize_digits else word
            pos = PAD_POS # CAUTION: POS will not be used now
            head = int(tokens[1])
            type = tokens[2]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            char_seqs.append([char for char in word])
            char_id_seqs.append([self.__char_alphabet.get_index(char) for char in word])

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos)) # CAUTION: POS will not be used not

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))

            morph_seqs.append([END_MORPH, ])
            morph_id_seqs.append([self.__morph_alphabet.get_index(END_MORPH), ])

            syll_seqs.append([END_SYLL, ])
            syll_id_seqs.append([self.__syll_alphabet.get_index(END_SYLL), ])

            morph_tag_seqs.append([END_MORPH_TAG, ])
            morph_tag_id_seqs.append([self.__morph_tag_alphabet.get_index(END_MORPH_TAG), ])

            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])

            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))

            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))

            heads.append(0)

        # Debugging
        # if words[1] != u'0000/00/00':
        #     print('words:', words)
        #     print('word_ids:', word_ids)
        #     print('morph_seqs:', morph_seqs)
        #     print('morph_id_seqs:', morph_id_seqs)
        #     print('syll_seqs:', syll_seqs)
        #     print('syll_id_seqs:', syll_id_seqs)
        #     print('morph_tag_seqs:', morph_tag_seqs)
        #     print('morph_tag_id_seqs:', morph_tag_id_seqs)
        #     print('char_seqs:', char_seqs)
        #     print('char_id_seqs:', char_id_seqs)
        #     print('postags:', postags)
        #     print('pos_ids:', pos_ids)
        #     print('types:', types)
        #     print('type_ids:', type_ids)
        #     print('heads:', heads)
        #
        #     exit()

        inst =  DependencyInstance(
            KrSentence(words, word_ids, morph_seqs, morph_id_seqs, syll_seqs, syll_id_seqs, morph_tag_seqs, morph_tag_id_seqs, char_seqs, char_id_seqs),
            postags, pos_ids, heads, types, type_ids)

        return inst

class RawSejongReader(SejongReader):
    def __init__(self, file_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        # CAUTION: currently using char_alphabet as morph_alphabet (morph/tag)
        super(RawSejongReader, self).__init__(file_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        self.__source_file = codecs.open(file_path, 'r', encoding='cp949')
        self.__word_alphabet = word_alphabet
        self.__morph_alphabet = morph_alphabet
        self.__syll_alphabet = syll_alphabet
        self.__morph_tag_alphabet = morph_tag_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        line = self.__source_file.readline()
        # skip multiple blank lines and comments.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        if line[0] == ';':
            raw_sent = line[2:].strip()
            raw_words = utils.split_raw(raw_sent)
            raw_morph_dict = utils.get_mor_result_v2(raw_sent)
        else:
            raise IOError('malformed file:', self.__source_file)

        words = []
        word_ids = []
        morph_seqs = []
        morph_id_seqs = []
        syll_seqs = []
        syll_id_seqs = []
        morph_tag_seqs = []
        morph_tag_id_seqs = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))

            morph_seqs.append([ROOT_MORPH, ])
            morph_id_seqs.append([self.__morph_alphabet.get_index(ROOT_MORPH), ])

            syll_seqs.append([ROOT_SYLL, ])
            syll_id_seqs.append([self.__syll_alphabet.get_index(ROOT_SYLL), ])

            morph_tag_seqs.append([ROOT_MORPH_TAG, ])
            morph_tag_id_seqs.append([self.__morph_tag_alphabet.get_index(ROOT_MORPH_TAG), ])

            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])

            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))

            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))

            heads.append(0)

        for word in raw_words:
            morphs_text = []
            morphs = []
            morph_ids = []
            sylls = []
            syll_ids = []
            morph_tags = []
            morph_tag_ids = []

            if raw_morph_dict is None or word not in raw_morph_dict:
                morphs.append(PAD_MORPH)
                morph_ids.append(self.__morph_alphabet.get_index(PAD_MORPH))
                sylls.append(PAD_SYLL)
                syll_ids.append(self.__syll_alphabet.get_index(PAD_SYLL))
                morph_tags.append(PAD_MORPH_TAG)
                morph_tag_ids.append(self.__morph_tag_alphabet.get_index(PAD_MORPH_TAG))
            else:
                for morph_d in raw_morph_dict[word]:
                    morphs_tags_list = morph_d.items()
                    morph_text, morph_tag = morphs_tags_list[0]
                    morph = utils.get_morphs(morphs_tags_list)[0]
                    syll = utils.get_sylls(morphs_tags_list)

                    if normalize_digits:
                        morph = utils.DIGIT_RE.sub(b"0", morph)
                        syll = [utils.DIGIT_RE.sub(b"0", s) for s in syll]

                    morphs.append(morph)
                    morph_ids.append(self.__morph_alphabet.get_index(morph))
                    sylls += syll
                    syll_ids += [self.__syll_alphabet.get_index(s) for s in syll]
                    morph_tags.append(morph_tag)
                    morph_tag_ids.append(self.__morph_tag_alphabet.get_index(morph_tag))

            if len(morphs) > utils.MAX_MORPH_LENGTH:
                morphs = morphs[:utils.MAX_MORPH_LENGTH]
                morph_ids = morph_ids[:utils.MAX_MORPH_LENGTH]
            morph_seqs.append(morphs)
            morph_id_seqs.append(morph_ids)

            if len(sylls) > utils.MAX_SYLL_LENGTH:
                sylls = sylls[:utils.MAX_SYLL_LENGTH]
                syll_ids = syll_ids[:utils.MAX_SYLL_LENGTH]
            syll_seqs.append(sylls)
            syll_id_seqs.append(syll_ids)

            if len(morph_tags) > utils.MAX_MORPH_LENGTH:
                morph_tags = morph_tags[:utils.MAX_MORPH_LENGTH]
                morph_tag_ids = morph_tag_ids[:utils.MAX_MORPH_LENGTH]
            morph_tag_seqs.append(morph_tags)
            morph_tag_id_seqs.append(morph_tag_ids)

            word = utils.DIGIT_RE.sub(b"0", word) if normalize_digits else word
            pos = PAD_POS  # CAUTION: POS will not be used now
            head = 0 # UNKNOWN in test file
            type = ROOT_TYPE # UNKNOWN in test file

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            char_seqs.append([char for char in word])
            char_id_seqs.append([self.__char_alphabet.get_index(char) for char in word])

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))  # CAUTION: POS will not be used not

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))

            morph_seqs.append([END_MORPH, ])
            morph_id_seqs.append([self.__morph_alphabet.get_index(END_MORPH), ])

            syll_seqs.append([END_SYLL, ])
            syll_id_seqs.append([self.__syll_alphabet.get_index(END_SYLL), ])

            morph_tag_seqs.append([END_MORPH_TAG, ])
            morph_tag_id_seqs.append([self.__morph_tag_alphabet.get_index(END_MORPH_TAG), ])

            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])

            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))

            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))

            heads.append(0)

        inst = DependencyInstance(
            KrSentence(words, word_ids, morph_seqs, morph_id_seqs, syll_seqs, syll_id_seqs, morph_tag_seqs,
                       morph_tag_id_seqs, char_seqs, char_id_seqs),
            postags, pos_ids, heads, types, type_ids)

        return inst

# TODO CoNLLXKrReader
class CoNLLXKrReader(CoNLLXReader):
    def __init__(self, file_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        super(CoNLLXKrReader, self).__init__(file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        self.__source_file = codecs.open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__morph_alphabet = morph_alphabet
        self.__syll_alphabet = syll_alphabet
        self.__morph_tag_alphabet = morph_tag_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            line = line.decode('utf-8')
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        morph_seqs = []
        morph_id_seqs = []
        syll_seqs = []
        syll_id_seqs = []
        morph_tag_seqs = []
        morph_tag_id_seqs = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))

            morph_seqs.append([ROOT_MORPH, ])
            morph_id_seqs.append([self.__morph_alphabet.get_index(ROOT_MORPH), ])

            syll_seqs.append([ROOT_SYLL, ])
            syll_id_seqs.append([self.__syll_alphabet.get_index(ROOT_SYLL), ])

            morph_tag_seqs.append([ROOT_MORPH_TAG, ])
            morph_tag_id_seqs.append([self.__morph_tag_alphabet.get_index(ROOT_MORPH_TAG), ])

            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])

            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))

            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))

            heads.append(0)

        for tokens in lines:
            word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
            morph_text = utils.split_plus(tokens[2])
            morph_tag = utils.split_plus(tokens[4])
            pos = PAD_POS # not used
            head = int(tokens[6])
            type = tokens[7]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, heads, types, type_ids)

