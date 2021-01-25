import sys
import shutil
import os

sys.path.append("./lib")

mode =  "FastLSTM"
num_epochs =  1000 
batch_size =  32 
decoder_input_size =  256 
hidden_size =  512 
encoder_layers =  3 
decoder_layers =  1

 
word_dim = 300 
morph_dim = 100 
syll_dim = 100 
morph_tag_dim = 50 
char_dim = 50 
 
num_filters = 100 
arc_space = 512 
type_space = 128 
 
opt = "adam" 
learning_rate = 0.001 
decay_rate = 0.75 
epsilon = 1e-4 
coverage = 0.0 
gamma = 0.0 
clip = 5.0 
 
schedule = 20 
double_schedule_decay = 5 
 
p_in = 0.33 
p_out = 0.33 
p_rnn = (0.33, 0.33)
unk_replace = 0.0 
label_smooth = 1.0 
beam = 1 
prior_order = "inside_out "
  
word_embedding = "random"
morph_embedding = "glove"
syll_embedding = "glove" 
 
word_path = "glove/MORPH_GLOVE" 
morph_path = "glove/MORPH_GLOVE" 
 
syll_path = "glove/SYLL_GLOVE" 
 
punctuation = '.' '``' "''" ':' ',' 
 
train_path = "data/UCorpus_DP_SR" 
dev_path = "data/UCorpus_DP_SR" 
test_path = "data/UCorpus_DP_SR" 
 
model_path = "models/MODEL_PATH/" 
model_name = "network.pt"

momentum = 0.9
betas = (0.9, 0.9)

word_dict = 'random'
morph_dict = 'random'
syll_dict = 'random'

