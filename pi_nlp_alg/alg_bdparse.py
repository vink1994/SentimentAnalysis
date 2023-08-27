import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
import argparse
import json

def parse_experiments(path):


    configs = []

    opt = argparse.ArgumentParser()
    with open(path, "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    for id, config in json_config.items():
        parser = argparse.ArgumentParser()
        configs.append(parser.parse_args())
    return configs

def mbert_datfeatex(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Domainknowledge(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[len(tokens)-300:len(tokens)], dtype='float32')
    return word_vec


def mbert_datfeat(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('word vectors loading')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.840B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('creating embedded mat', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def mbert_mat_gen(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Domainknowledge(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def mbert_TextSent_gen(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return mbert_mat_gen(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class MBERT_DKnldg:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def mbert_TextSent_gen(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return mbert_mat_gen(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class MBERT_Parsefunc(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        print('creating indices......')
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polarity = int(polarity) + 1
            aspect_indices = tokenizer.mbert_TextSent_gen(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            text_left = ' '.join(text_left.split(' ')[int(-(tokenizer.max_seq_len-aspect_len)/2-5):])
            text_right = ' '.join(text_right.split(' ')[:int((tokenizer.max_seq_len-aspect_len)/2-5)])
            text_raw = text_left +' '+ aspect +' '+ text_right
            text_raw_indices = tokenizer.mbert_TextSent_gen(text_raw)

            text_bert_indices = tokenizer.mbert_TextSent_gen('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = mbert_mat_gen(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.mbert_TextSent_gen("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.mbert_TextSent_gen("[CLS] " + aspect + " [SEP]")


            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_raw_indices': text_raw_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
            }

            all_data.append(data)

        self.data = all_data


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

