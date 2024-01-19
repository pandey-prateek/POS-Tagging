import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as  vocab
import pyconll 
import string
from sklearn.metrics import accuracy_score,f1_score,classification_report

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size,layers):
        super(LSTMTagger, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True,num_layers=layers)
        self.hidden2tag = nn.Linear(2*hidden_dim, target_size)
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def get_data_from_file(file):
    data_file = pyconll.load_from_file(file)
    sentences=[]
    pos_tags=[]
    all_tags=[]
    for data in data_file:
        sentence=[]
        pos=[]
        
        for token in data:
            sentence.append(token.form.lower())
            pos.append(token.upos)
            
        pos_tags.append(pos)
        sentences.append(sentence)
        all_tags+=pos
    return sentences,pos_tags,all_tags
def yield_tokens(tokenlist):
    for tokens in tokenlist:
        yield tokens
def get_data(dataset, vocab):
    data = []
    for sentence in dataset:
            tokens = [vocab[token] for token in sentence]
            data.append(torch.LongTensor(tokens))
    return data
sentences,pos_tags,all_tags=get_data_from_file('./UD_English-Atis/en_atis-ud-train.conllu')
word_vocab=vocab.build_vocab_from_iterator(yield_tokens(sentences),min_freq=1,specials=["<UNK>"])
word_vocab.set_default_index(word_vocab["<UNK>"])
data=get_data(sentences,word_vocab)

tag_vocab=vocab.build_vocab_from_iterator(yield_tokens(pos_tags),min_freq=1,specials=["<UNK_TAG>"])
tag_vocab.set_default_index(tag_vocab["<UNK_TAG>"]) 
tag_data=get_data(pos_tags,tag_vocab) 

sentences_val,pos_tags_val,all_val_tags=get_data_from_file('./UD_English-Atis/en_atis-ud-dev.conllu')
val_data=get_data(sentences_val,word_vocab)
val_tag=get_data(pos_tags_val,tag_vocab)

sentences_test,pos_tags_test,all_test_tags=get_data_from_file('./UD_English-Atis/en_atis-ud-test.conllu')
test_data=get_data(sentences_test,word_vocab)
test_tag=get_data(pos_tags_test,tag_vocab)

EMBEDDING_DIM = 64
HIDDEN_DIM = 64
EPOCHS = 30
LEARNING_RATE=0.1
NUMBER_OF_LAYERS=1
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,word_vocab.__len__(), tag_vocab.__len__(),NUMBER_OF_LAYERS)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
s=input("ENTER SENTENCE ")
s=s.lower()
s = s.translate(str.maketrans('', '', string.punctuation))
s= s.strip()
saved=True

if saved:
    model.load_state_dict(torch.load('pos_tagging_lstm.pt'))
    s=[s.split(' ')]
    test_data=get_data(s,word_vocab)
    with torch.no_grad():
        for i in range(len(s)):
            
            tag_scores = model(test_data[i])
            indices = torch.max(tag_scores, 1)[1]
            ret = []
            
            for j in range(len(indices)):
                for key, value in tag_vocab.get_stoi().items():
                    if indices[j] == value:
                        ret.append((s[i][j], key))
            for val in ret:
                print(val[0]+"\t"+val[1])
else:
    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):
        for i in range(len(sentences)):
            model.zero_grad()
            tag_scores = model(data[i])
            loss = loss_function(tag_scores, tag_data[i])
            indices = torch.max(tag_scores, 1)[1]
            loss.backward()
            optimizer.step()
        
        valid_loss=0
        acc=0
        tags=[]
        with torch.no_grad():
            for i in range(len(sentences_val)):
                
                tag_scores = model(val_data[i])
                loss = loss_function(tag_scores, val_tag[i])
                valid_loss+=loss.item()
                indices = torch.max(tag_scores, 1)[1]
                ret = []
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'pos_tagging_lstm.pt')    
    tags=[]
    with torch.no_grad():
        for i in range(len(sentences_test)):
            
            tag_scores = model(test_data[i])
            loss = loss_function(tag_scores, test_tag[i])
            indices = torch.max(tag_scores, 1)[1]
            for j in range(len(indices)):
                for key, value in tag_vocab.get_stoi().items():
                    if indices[j] == value:
                        tags.append(key)
        print(classification_report(all_test_tags,tags))
    s=[s.split(' ')]
    test_data=get_data(s,word_vocab)
    with torch.no_grad():
        for i in range(len(s)):
            
            tag_scores = model(test_data[i])
            indices = torch.max(tag_scores, 1)[1]
            ret = []
            
            for j in range(len(indices)):
                for key, value in tag_vocab.get_stoi().items():
                    if indices[j] == value:
                        ret.append((s[i][j], key))
            print('\n')
            print(ret)        

            # print(ret)