from flask import request
import logging
import flask

import numpy as np
import torch 
import argparse
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertForSequenceClassification, BertTokenizer
import logging
logger = logging.getLogger(__name__)
import collections

import spacy
import en_core_web_sm #python -m spacy download en_core_web_sm
nlp = en_core_web_sm.load()

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

from collections import defaultdict

app = flask.Flask(__name__)


class SensesVSM(object):

    def __init__(self, vecs_path, normalize=True):
        self.vecs_path = vecs_path
        self.labels = []
        self.vectors = np.array([], dtype=np.float32)
        self.indices = {}
        self.ndims = 0

        if self.vecs_path.endswith('.txt'):
            self.load_txt(self.vecs_path)

        elif self.vecs_path.endswith('.npz'):
            self.load_npz(self.vecs_path)

        self.load_aux_senses()

        if normalize:
            self.normalize()

    def load_txt(self, txt_vecs_path):
        self.vectors = []
        with open(txt_vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.split()
                self.labels.append(elems[0])
                self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))
        self.vectors = np.vstack(self.vectors)

        self.labels_set = set(self.labels)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def load_npz(self, npz_vecs_path):
        loader = np.load(npz_vecs_path)
        self.labels = loader['labels'].tolist()
        self.vectors = loader['vectors']

        self.labels_set = set(self.labels)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def load_aux_senses(self):

        self.sk_lemmas = {sk: get_sk_lemma(sk) for sk in self.labels}
        self.sk_postags = {sk: get_sk_pos(sk) for sk in self.labels}

        self.lemma_sks = collections.defaultdict(list)
        for sk, lemma in self.sk_lemmas.items():
            self.lemma_sks[lemma].append(sk)
        self.known_lemmas = set(self.lemma_sks.keys())

        self.sks_by_pos = collections.defaultdict(list)
        for s in self.labels:
            self.sks_by_pos[self.sk_postags[s]].append(s)
        self.known_postags = set(self.sks_by_pos.keys())

    def save_npz(self):
        npz_path = self.vecs_path.replace('.txt', '.npz')
        np.savez_compressed(npz_path,
                            labels=self.labels,
                            vectors=self.vectors)

    def normalize(self, norm='l2'):
        norms = np.linalg.norm(self.vectors, axis=1)
        self.vectors = (self.vectors.T / norms).T

    def get_vec(self, label):
        return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def match_senses(self, vec, lemma=None, postag=None, topn=100):

        relevant_sks = []
        for sk in self.labels:
            if (lemma is None) or (self.sk_lemmas[sk] == lemma):
                if (postag is None) or (self.sk_postags[sk] == postag):
                    relevant_sks.append(sk)
        relevant_sks_idxs = [self.indices[sk] for sk in relevant_sks]

        sims = np.dot(self.vectors[relevant_sks_idxs], np.array(vec))
        matches = list(zip(relevant_sks, sims))

        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        return matches[:topn]

    def most_similar_vec(self, vec, topn=10):
        sims = np.dot(self.vectors, vec).astype(np.float32)
        sims_ = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            r.append((self.labels[top_i], sims_[top_i]))
        return r

    def sims(self, vec):
        return np.dot(self.vectors, np.array(vec)).tolist()




def get_sk_type(sensekey):
    return int(sensekey.split('%')[1].split(':')[0])


def get_sk_pos(sk, tagtype='long'):
    # merges ADJ with ADJ_SAT

    if tagtype == 'long':
        type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
        return type2pos[get_sk_type(sk)]

    elif tagtype == 'short':
        type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
        return type2pos[get_sk_type(sk)]


def get_sk_lemma(sensekey):
    return sensekey.split('%')[0]


def get_synset_offset_definition(synset_key):
        
        lemma = synset_key.split('%')[0]

        for synset in wn.synsets(lemma):
            for lemma in synset.lemmas():

                if synset_key == lemma.key():
                
                    return synset, synset.offset(), synset.definition()
        return None



def get_sent_info(sentence, merge_ents=False):
    
    sent_info = {'tokens': [], 'lemmas': [], 'pos': [], 'sentence': ''}

    sent_info['sentence'] = sentence
    doc = nlp(sent_info['sentence'])

    if merge_ents:
        for ent in doc.ents:
            ent.merge()

    for tok in doc:
        
        sent_info['tokens'].append(tok.text.replace(' ', '_'))
        sent_info['lemmas'].append(tok.lemma_)
        sent_info['pos'].append(tok.pos_)

    sent_info['tokenized_sentence'] = ' '.join(sent_info['tokens'])

    return sent_info


def map_senses(svsm,model, tokenizer, word_detail, tokens, postags=[], lemmas=[], use_postag=False, use_lemma=False):

    marked_text = "[CLS] " + ' '.join(tokens) + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)
    
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    # FOR GPU

    #tokens_tensor = torch.tensor([indexed_tokens]).cuda()    
    #segments_tensors = torch.tensor([segments_ids]).cuda()    
    
    with torch.no_grad():
        
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[1]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs_sum = []
    
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)

    token_vecs_sum.pop(0)
    token_vecs_sum.pop(-1)
        
    sent_bert = token_vecs_sum
    
    #sent_bert = [tensor.cpu() for tensor in sent_bert]  #TAKE OUTPUT GPU TENSORS BACK TO CPU AFTER PROCESSING
    
    matches = []

    if len(tokens) != len(postags):  
        use_postag = False

    if len(tokens) != len(lemmas):  
        use_lemma = False

    
    index_list = []
    
    for check in word_detail:

        if check in tokens:
            indexes = [i for i, x in enumerate(tokens) if x == check]
            index_list.append(indexes)

    word_ind = [item for sublist in index_list for item in sublist]

    for index_num in word_ind:
        
        idx_vec = sent_bert[index_num]
        idx_vec = idx_vec / np.linalg.norm(idx_vec)

        if svsm.ndims == 1024:
            pass

        elif svsm.ndims == 1024+1024:
            idx_vec = np.hstack((idx_vec, idx_vec))
            idx_vec = idx_vec / np.linalg.norm(idx_vec)

        idx_matches = []
        
        if use_lemma and use_postag:
            idx_matches = svsm.match_senses(idx_vec, lemmas[idx], postags[idx], topn=None)

        elif use_lemma:
            idx_matches = svsm.match_senses(idx_vec, lemmas[idx], None, topn=None)

        elif use_postag:
            idx_matches = svsm.match_senses(idx_vec, None, postags[idx], topn=None)

        else:
            idx_matches = svsm.match_senses(idx_vec, None, None, topn=1)

        matches.append(idx_matches)
         

    return matches, word_ind, tokens

BERT_BASE_DIR = 'bert_torch_model/'    
vec_path = 'lmms_1024.bert-large-cased.npz'

model = BertForSequenceClassification.from_pretrained(BERT_BASE_DIR, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(BERT_BASE_DIR, do_lower_case=True)

# To load bert model in GPU

#model = model.cuda()

model.eval()

senses_vsm = SensesVSM(vec_path, normalize=True)

@app.route("/synset_processing", methods=['POST'])

def predict_synset():


    data = request.json
    print('requested_data', data)
    
    sentence = data['sentence']
    word_detail = data['word']
    

# TO CHECK IF GPU IS AVAILABLE

#     if torch.cuda.is_available(): 
#     print('GPU DEVICE IS AVAILABLE')
#     print(f' ID of the device is : {torch.cuda.current_device()}')
    
    sent_info = get_sent_info(sentence)
    
    matches, word_indexes, tokens = map_senses(senses_vsm, model, tokenizer, word_detail, sent_info['tokens'],
                                     sent_info['pos'],
                                     sent_info['lemmas'],
                                     use_lemma=False,
                                     use_postag=False)

    matches = [list(items[0]) for items in matches]

    words = [tokens[items] for items in word_indexes]

    result = []

    for idx_matches in zip(matches, words):

        synset_key = idx_matches[0][0]
        word = idx_matches[1]
        synset, offset, definition = get_synset_offset_definition(synset_key)
        
        result.append({'word':word,'synset':str(synset),'synset_key':synset_key,'offset':offset,'definition':definition})
                
    out = {'bert_WSD': result}
    return out

if __name__ == "__main__":
      app.run(host='0.0.0.0',debug=True, use_reloader=False)