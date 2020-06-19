import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, inputs):
        return self.embedding(inputs)
    

class Position_wise_FFN(nn.Module):
    def __init__(self):
        super(Position_wise_FFN, self).__init__()
        self.linear_1 = nn.Linear()
        self.relu = F.relu()
        self.linear_2 = nn.Linear()

    def forward(self, inputs):
        linear_1 = self.linear_1(inputs)
        relu = self.relu(linear_1)
        linear_2 = self.linear_2(relu)

        return linear_2


class Scaled_Dot_Product(nn.Module):
    def __init__(self, d_head):
        super(Scaled_Dot_Product, self).__init__()
        self.sqrt = 1 / d_head**0.5

    def forward(self, query, key, value, masked):
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = torch.matmul(scores, self.sqrt)
        scores.masked_fill_(masked, -1e9)

        softmax = F.softmax(scores, dim=-1)
        output = torch.matmul(softmax, value)

        return output


class Multi_Head_Attention(nn.Module):
    def __init__(self, embed_dim, head, dropout):
        super(Multi_Head_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.head = head
        self.dk = embed_dim / head
        self.dv = embed_dim / head

        self.query_linear = nn.Linear(self.embed_dim, self.dk, bias=False)
        self.key_linear = nn.Linear(self.embed_dim, self.dk, bias=False)
        self.value_linear = nn.Linear(self.embed_dim, self.dv, bias=False)
        self.scaled_dot_product = Scaled_Dot_Product(self.dv)
        self.output_linear = nn.Linear(self.dv, self.embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, masked):
        batch_size = query.size(0)

        query = self.query_linear(query).view(batch_size, -1, self.head, self.dk).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.head, self.dk).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.head, self.dv).transpose(1, 2)
        masked = masked.unsqueeze(1).repeat(1, self.head, 1, 1)

        scores = self.scaled_dot_product(query, key, value, masked)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.output_linear(scores)

        return output


class Encoder(nn.Module):
    def __init__(self, n):
        super(Encoder, self).__init__()

    def forward(self, *input):
        pass


class Decoder(nn.Module):
    def __init__(self, n):
        super(Decoder, self).__init__()

    def forward(self, *input):
        pass


class Transformer(BaseModel):
    def __init__(self, n):
        super(Transformer, self).__init__()

    def forward(self, *inputs):
        pass
