import argparse

#1 standard language input
#sequence of vector or cnn of vector
#2 generate language 
# seeds in ct, random input.
#3 discriminator
#cnn sigmoid 

def gan():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    
    opt = parser.parse_args()
    print(opt)
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    return

#gan()

#-*- coding:utf-8 -*-
from keras.layers.core import Activation, Dense, Dropout, RepeatVector, Lambda, Reshape
from keras.layers import merge, Input, Merge
#from keras.regularizers import l1, l2, activity_l2, l1l2
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras.models import Sequential, Model
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from numpy import hstack
import h5py
import sys
import numpy as np
from keras.layers.convolutional import Convolution1D
from keras.utils.np_utils import to_categorical

default_encoding = "utf-8"
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

MAX_LEN = 25
WORD_EMBEDDING_LEN = 300
SENTENCE_EMBEDDING_LEN = 300

#1. Process Text -- Word Embedding using w2v, processed, stored in NPY fils
sample_path="./data/cmb_train.npy"
sample_label_path="./data/cmb_label.npy"
sample_feature_path="./data/cmb_train_feature.npy"
test_path="./data/cmb_test.npy"
test_label_path="./data/cmb_test_label.npy"
test_feature_path="./data/cmb_test_feature.npy"
##1.load data
sample_vector=np.load(sample_path)
sample_feature_vector=np.load(sample_feature_path)
sample_label_vector=to_categorical(np.load(sample_label_path), 20)
print(np.shape(sample_vector))

test_vector=np.load(test_path)
test_feature_vector=np.load(test_feature_path)
test_label_vector=to_categorical(np.load(test_label_path), 20)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        nn.GRU()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

##2.get training data

# #2. LSTM for Q and A -- condition on Q
# input_a = Input(shape=(MAX_LEN, WORD_EMBEDDING_LEN), name='input_a')
# input_b = Input(shape=(3,),name='input_b')
input_a = Variable(torch.from_numpy(sample_vector[0:127,:,:])).float()
input_b = Variable(torch.from_numpy(sample_feature_vector)).float()
target = Variable(torch.from_numpy(sample_label_vector)).float()

#tower_a_0 = Convolution1D(SENTENCE_EMBEDDING_LEN, 1, border_mode='same', activation='relu')(input_a)
# tower_a_1 = Convolution1D(SENTENCE_EMBEDDING_LEN, 2, border_mode='same', activation='relu')(tower_a_0)
# tower_a_2 = Convolution1D(SENTENCE_EMBEDDING_LEN, 3, border_mode='same', activation='relu')(tower_a_0)
# tower_a_3 = Convolution1D(SENTENCE_EMBEDDING_LEN, 4, border_mode='same', activation='relu')(tower_a_0)

tower_a_0 = F.relu(nn.Conv1d(MAX_LEN,MAX_LEN, kernel_size=1,stride=1,  padding=0)(input_a))
# tower_a_1 = Convolution1D(SENTENCE_EMBEDDING_LEN, 2, border_mode='same', activation='relu')(tower_a_0)
# tower_a_2 = Convolution1D(SENTENCE_EMBEDDING_LEN, 3, border_mode='same', activation='relu')(tower_a_0)
# tower_a_3 = Convolution1D(SENTENCE_EMBEDDING_LEN, 4, border_mode='same', activation='relu')(tower_a_0)
print(tower_a_0)
# merged_0 =  merge([tower_a_1,tower_a_2, tower_a_3], mode='concat', concat_axis=-1)
# forward_a_1 = GRU(SENTENCE_EMBEDDING_LEN, 
#     go_backwards=True,
#     return_sequences=False)(merged_0)
# backward_a_1 = GRU(SENTENCE_EMBEDDING_LEN,
#     go_backwards=False,
#     return_sequences=False)(merged_0)
# 
# 
# merged = merge([forward_a_1,backward_a_1, input_b], mode='concat', concat_axis=-1)
# hidden_1 = Dense(128)(merged)
# act_1 = ELU()(hidden_1)
# after_dp_2 = Dropout(0.9)(act_1)
# 
# output = Dense(20, activation='softmax')(after_dp_2)
# #5. compile
# model = Model(input=[input_a, input_b], output=output)
# model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
# 
# print(model.summary())
# 
# 
# callback = ModelCheckpoint('./model/model_cmb_act_weights_sub.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
# model.fit([sample_vector, sample_feature_vector], sample_label_vector,
#                 batch_size=128, nb_epoch=2000,
#                 callbacks=[callback],
#                 validation_data=([test_vector, test_feature_vector], test_label_vector))