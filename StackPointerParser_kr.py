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

	doc, tags, dep_tuples_list = read_ucorpus_all(train_path)
	print(len(doc))

	if word_embedding != 'random':
		word_dict, word_dim = load_embedding_dict(word_embedding, word_path)
	
	print(word_dict)
	# if morph_embedding != 'random':
	# 	morph_dict, morph_dim = load_embedding_dict(morph_embedding, morph_path)
	# if syll_embedding != 'random':
	# 	syll_dict, syll_dim = load_embedding_dict(syll_embedding, syll_path)


	logger.info("Creating Alphabets")
	alphabet_path = os.path.join(model_path, 'alphabets/')

	model_name = "network.pt"
	print(model_name)
	model_name = os.path.join(model_path, model_name)
	word_alphabet, char_alphabet, pos_alphabet, type_alphabet = create_alphabets(alphabet_path, train_path, data_paths=[dev_path, test_path],
																										max_vocabulary_size=50000, embedd_dict=word_dict)
	# create_alphabets(alphabet_path, train_path, data_paths=[dev_path, test_path], max_vocabulary_size=50000, embedd_dict=word_dict) 


	# end	
	# end	
	# end	
	# end	
	# end	


if __name__ == "__main__" :
	main()
