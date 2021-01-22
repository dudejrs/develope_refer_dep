__author__ = 'char'


import codecs
from neuronlp2.io.writer import CoNLLXWriter


class SejongWriter(CoNLLXWriter):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, raw_file=None):
        super(SejongWriter, self).__init__(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet
        self.__raw_lines = None
        self.__raw_line_idx = 0

        if raw_file:
            self.__raw_lines = []
            with codecs.open(raw_file, 'r', encoding='cp949') as f:
                for line in f:
                    if line[0] == ';':
                        self.__raw_lines.append(line)

    def start(self, file_path):
        self.__source_file = codecs.open(file_path, 'w', encoding='cp949')

    def start_strio(self, strio):
        self.__source_file = strio

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, head, type, lengths, symbolic_root=False, symbolic_end=False, ignore_word=True, raw_words=None):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            if self.__raw_lines:
                self.__source_file.write(self.__raw_lines[self.__raw_line_idx])
                self.__raw_line_idx += 1

            for j in range(start, lengths[i] - end):
                if raw_words and j > 0:
                    w = raw_words[i][j-1]
                else:
                    w = self.__word_alphabet.get_instance(word[i, j])

                # p = self.__pos_alphabet.get_instance(pos[i, j])
                t = self.__type_alphabet.get_instance(type[i, j])
                h = head[i, j]
                if ignore_word:
                    self.__source_file.write('%d\t%d\t%s\n' % (j, h, t))
                else:
                    self.__source_file.write('%d\t%d\t%s\t%s\n' % (j, h, t, w))
            self.__source_file.write('\n')
