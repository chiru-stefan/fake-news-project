import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
from IPython.display import clear_output
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch
from keras.preprocessing.sequence import pad_sequences


class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens, masks=None):
        pooled_output = self.bert(tokens, attention_mask=masks, output_hidden_states=False)[1]
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba


def preprocess_text_inference(sentences, tokenizer='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(tokenizer, do_lower_case=True)

    tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], [sentences]))

    tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, tokens)), maxlen=512,
                               truncating="post", padding="post", dtype="int")
    tokens_tensor = torch.tensor(tokens_ids)
    masks = [[float(i > 0) for i in ii] for ii in tokens_tensor]
    masks_tensor = torch.tensor(masks)
    return tokens_tensor, masks_tensor

def test_dataset_tokenized_sentences(sentences, labels, tokenizer='bert-base-uncased'):
    '''
    Takes in a list of sentences and labels and returns a tensor dataset
    '''
    tokenizer = BertTokenizer.from_pretrained(tokenizer, do_lower_case=True)

    tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], sentences))
    y_tensor = torch.tensor(labels.reshape(-1, 1)).float()

    tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, tokens)), maxlen=512,
                                     truncating="post", padding="post", dtype="int")
    tokens_tensor = torch.tensor(tokens_ids)
    masks = [[float(i > 0) for i in ii] for ii in tokens_tensor]
    masks_tensor = torch.tensor(masks)

    test_dataset = TensorDataset(tokens_tensor, masks_tensor, y_tensor)

    return test_dataset


def dataset_tokenized_sentences(sentences, labels, test_only=True, tokenizer='bert-base-uncased'):
    '''
    Takes in a list of sentences and labels and returns a tensor dataset
    '''
    tokenizer = BertTokenizer.from_pretrained(tokenizer, do_lower_case=True)

    if test_only == False:
        dataset_out = sentences

        # split into train and test
        training_data = dataset_out[:int(len(dataset_out) * 0.8)]
        testing_data = dataset_out[int(len(dataset_out) * 0.8):]
        train_tokens = list(
                        map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], training_data.data.values))
        test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], testing_data.data.values))
        train_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, train_tokens)), maxlen=512,
                                         truncating="post", padding="post", dtype="int")
        test_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, test_tokens)), maxlen=512,
                                        truncating="post", padding="post", dtype="int")

        train_y = training_data.label.astype(np.float32).values
        test_y = testing_data.label.astype(np.float32).values
        train_y.shape, test_y.shape, np.mean(train_y), np.mean(test_y)

        train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
        test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

        train_tokens_tensor = torch.tensor(train_tokens_ids)
        train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()

        test_tokens_tensor = torch.tensor(test_tokens_ids)
        test_y_tensor = torch.tensor(test_y.reshape(-1, 1)).float()

        train_masks_tensor = torch.tensor(train_masks)
        test_masks_tensor = torch.tensor(test_masks)

        train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

        test_dataset = TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

        return train_dataloader, test_dataloader
    else:
        tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], sentences))
        y_tensor = torch.tensor(labels.reshape(-1, 1)).float()

        tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, tokens)), maxlen=512,
                                         truncating="post", padding="post", dtype="int")
        tokens_tensor = torch.tensor(tokens_ids)
        masks = [[float(i > 0) for i in ii] for ii in tokens_tensor]
        masks_tensor = torch.tensor(masks)

        test_dataset = TensorDataset(tokens_tensor, masks_tensor, y_tensor)

        return test_dataset