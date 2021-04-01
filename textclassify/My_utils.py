# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000 #词表长度限制
UNK, PAD = '<UNK>', '<PAD>'


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



def build_vocab(file_path,tokenizer, max_size, min_freg):
    vocab_dic = {}
    with open(file_path,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freg], key=lambda x:x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK:len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    '''
    输入: config, ues_word
    输出:
    vocab, 词汇表
    train, 训练集，格式：[([...],0,32),([...],1,32),...]
    dev,   验证集，格式：[([...],0,32),([...],1,32),...]
    test,  测试集，格式：[([...],0,32),([...],1,32),...]
    '''
    if ues_word:
        tokenizer = lambda x: x.split(' ') # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x] # char-level

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb')) #读入词典
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer,max_size=MAX_VOCAB_SIZE,min_freg=1)
        pkl.dump(vocab, open(config.vocab_path,'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents


    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return vocab, train, dev, test

class DatasetIterater():
    '''
    build_iterator(train_data, config)
    train_data:[([...],0,32),([...],1,32),...]
    DataetIterater(dataset, config.batch_size, config.device)
    batches ---> train_data
    '''
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self,datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        return (x, seq_len),y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


