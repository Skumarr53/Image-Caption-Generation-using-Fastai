from  tqdm import tqdm
import numpy as np
import fasttext as ft
import json
import pickle
from pdb import set_trace

emb_path = '/home/skumar/.nlp_wordembeddings/cc.en.300.bin'
word_map_path = '/home/skumar/DataScience/Projects_Section/Projects_Working/Image_Captioning_Pytorch/flicker8k-dataset/Vocab_5_cap_per_img_2_min_word_freq.json' # word map index dict path
embExport_pkl_path = '/home/skumar/DataScience/Projects_Section/Projects_Working/Image_Captioning_Pytorch/flicker8k-dataset/Fastext_embedd_wordMap.pkl'
emb_dim = 300

# get words in the wordmap with index

with open(word_map_path,'r') as j:
    word_map = json.load(j)

# create a dictionary of words and corresponding verctor array
word_emb = np.zeros((len(word_map),emb_dim))
missing  = ['NA']*len(word_map)

# load fasttext word vectors
en_vecs = ft.load_model(str(emb_path))

for i,k in tqdm(enumerate(word_map)):
    if k in en_vecs.get_words():
        word_emb[i] = en_vecs.get_word_vector(k)
    else:
        missing[i] = k

print(missing)

with open(embExport_pkl_path,'wb') as f:
    pickle.dump(word_emb, f, 2)