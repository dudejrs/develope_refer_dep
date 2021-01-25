from __init__ import *

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD, Adamax

from neuronlp2.io import get_logger
from neuronlp2.models import StackPtrNet
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser

from neuronlp_kr.utils import load_embedding_dict
from neuronlp_kr.io.ulsan_data import create_alphabets
from lib.ucorpus_parser import read_ucorpus_all

logger = get_logger("PtrParser")

def main() :
	word_embedding = 'word2vec'
	word_path = "data/embedding/social_skipgram_test_w2v" 


	if word_embedding != 'random':
		word_dict, word_dim = load_embedding_dict(word_embedding, word_path)
	
	print(word_dict)
	# if morph_embedding != 'random':
	# 	morph_dict, morph_dim = load_embedding_dict(morph_embedding, morph_path)
	# if syll_embedding != 'random':
	# 	syll_dict, syll_dim = load_embedding_dict(syll_embedding, syll_path)



	# [Step 1]

	logger.info("Creating Alphabets")
	alphabet_path = os.path.join(model_path, 'alphabets/')

	model_name = "network.pt"
	print(model_name)
	model_name = os.path.join(model_path, model_name)
	word_alphabet, char_alphabet, pos_alphabet, type_alphabet = create_alphabets(alphabet_path, train_path, data_paths=[dev_path, test_path],
																										max_vocabulary_size=50000, embedd_dict=word_dict)

    num_words = word_alphabet.size()
    # num_morphs = morph_alphabet.size()
    # num_sylls = syll_alphabet.size()
    # num_morph_tags = morph_tag_alphabet.size()
    # num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    # num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    # logger.info("Morpheme Alphabet Size: %d" % num_morphs)
    # logger.info("Syllable Alphabet Size: %d" % num_sylls)
    # logger.info("Morpheme tag Alphabet Size: %d" % num_morph_tags)
    # logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    # logger.info("Type Alphabet Size: %d" % num_types)


    # [Step 2]

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    data_train = sejong_stacked_data.read_stacked_data_to_variable(train_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                                        use_gpu=use_gpu, prior_order=prior_order, data_format=data_format)
    num_data = sum(data_train[1])
    data_dev = sejong_stacked_data.read_stacked_data_to_variable(dev_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                                        use_gpu=use_gpu, volatile=True, prior_order=prior_order, data_format=data_format)
    data_test = sejong_stacked_data.read_stacked_data_to_variable(test_path, word_alphabet, morph_alphabet, syll_alphabet, morph_tag_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                                        use_gpu=use_gpu, volatile=True, prior_order=prior_order, data_format=data_format)



	# [TEST CODE]
	# shutil.rmtree(alphabet_path)


	# end	
	# end	
	# end	
	# end	
	# end	


if __name__ == "__main__" :
	main()
