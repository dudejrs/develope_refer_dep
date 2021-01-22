__author__ = 'cha'

from neuronlp2.io.instance import Sentence

class KrSentence(Sentence):
    def __init__(self, words, word_ids, morph_seqs, morph_id_seqs, syll_seqs, syll_id_seqs, morph_tag_seqs, morph_tag_id_seqs, char_seqs, char_id_seqs):
        super(KrSentence, self).__init__(words, word_ids, char_seqs, char_id_seqs)
        self.words = words
        self.word_ids = word_ids
        self.morph_seqs = morph_seqs
        self.morph_id_seqs = morph_id_seqs
        self.syll_seqs = syll_seqs
        self.syll_id_seqs = syll_id_seqs
        self.morph_tag_seqs = morph_tag_seqs
        self.morph_tag_id_seqs = morph_tag_id_seqs
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs

    def length(self):
        return len(self.words)