#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Config():
    '''
    配置参数
    '''
    def __init__(self, dataset, embedding):
        self.model_name = 'MyTextCNN'
        self.train_path  = dataset +'/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
                           dataset + '/data/class.txt',encoding = 'utf-8').readlines()]
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
                                    np.load(dataset + '/data/' + embedding)['embeddings'].astype('float32')) \
                                    if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5
        self.requite_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0     # 词表大小，在运行时赋值
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1)\
                     if self.embedding_pretrained is not None else 300  # 字向量维度
        self.filter_sizes = (2,3,4)

        self.num_filters = 256  # 卷积核数量(channels数)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        #如果存在预训练模型，则直接导入，否则，初始化词语的词向量
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx = config.n_vocab - 1)

        #构建网络
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # (1,256,(2,300))
        # (1,256,(3,300))
        # (1,256,(4,300))
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes]
        )
        # 设置dropout的值
        self.dropout = nn.Dropout(config.dropout)

        # 最后的全连接层
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
    def conv_and_pool(self, x ,conv):
        # 激活函数使用relu
        '''
        x.shape =  [batch_size,channels, seq_len,embed_dim]
        假设 batch_size = 3
        则，x.shape = (3, 1, 32, 300)
        经过conv2d的卷积，比如 conv2d(1,256,(2,300))
        得到的维度:
        [batch_size, out_channels, after_len, 1]
        即：
        (3, 256, 31, 1)
        '''
        # 将第四维度压缩掉，变成 [batch_size, out_channels, after_len]
        x = F.relu(conv(x)).squeeze(3)
        # 最大池化操作,求取序列中最大的数字，变成[batch_size, out_channels, 1],然后再压缩:[batch_size, out_channels]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    def forward(self, x):
        # 前向传播
        '''
        x = ([....],32)
        x[0] 表示一句话的词的索引列表，longtensor
        '''
        out = self.embedding(x[0])
        #print(out.shape)
        out = out.unsqueeze(1) # 在第二维度进行扩展, (32) --- > [32,1]
        # 求取每个维度的最大值，在第2维拼接起来，即纵轴拼接
        '''
        三个(2,3,4) 最后出来的都是 (batch_size, 236),设 batch_size = 3
        则3 个 (3,256),(3,256), (3,256) 拼接后:
        (3, 768)
        '''
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # 随机失活部分神经元
        out = self.dropout(out)
        # 全连接输出
        out = self.fc(out)
        # print('after softmax without dim:', without_dim_out.shape)
        # print('after softmax with dim=1:', dim_out.shape)
        return out





# 测试代码
# dataset = r'D:\workspace\programs\text_classify\textcnn\Chinese-Text-Classification\THUCNews'
# embedding = 'embedding_SougouNews.npz'
# config = Config(dataset , embedding)
# print(config.embed)
#
# tx = torch.LongTensor([i for i in range(1, 33)]).view(1,-1)
# x = (tx, 1)
# model = Model(config)
# print(model.forward(x))
