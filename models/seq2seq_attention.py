import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden):
        embedding = self.dropout(self.embedding(inputs))
        output, hidden = self.lstm(embedding, hidden)
        return output, hidden


class BahdanauDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers=1, dropout=0):
        super(BahdanauDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs, hidden, encoder_outputs):
        # Embedding input words
        embedding = self.dropout(self.embedding(inputs))

        # Calculating alignment score
        tanh = torch.tanh(self.fc_hidden(hidden[0]) + self.fc_encoder(encoder_outputs))
        alignment_score = tanh.bmm(self.weight.unsqueeze(2))

        # Softmaxing alignment score
        attention_weight = F.softmax(alignment_score, dim=1)

        # Multiplying attention weight with encoder outputs
        context_vector = torch.bmm(attention_weight.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # Concatenating context vector with embedded words
        output = torch.cat((embedding, context_vector[0]), 1).unsqueeze(0)

        # Passing LSTM
        output, hidden = self.lstm(output, hidden)
        output = F.log_softmax(self.classifier(output[0]), dim=1)

        return output, hidden, attention_weight

class LuongDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers=1, dropout=0, method="dot"):
        super(LuongDecoder, self).__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.method = method

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)

        if self.method == "general":
            self.general_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif self.method == "concat":
            self.concat_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_dim))

        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input, hidden, encoder_outputs):
        embedding = self.dropout(self.embedding(input))

        output, hidden = self.lstm(embedding, hidden)

        if self.method == "dot":
            alignment_score = encoder_outputs.bmm(output)
        elif self.method == "general":
            output = self.general_linear(output)
            alignment_score = encoder_outputs.bmm(output)
        elif self.method == "concat":
            output = torch.tanh(self.concat_linear(encoder_outputs + output))
            alignment_score = output.bmm(self.weight)

        attention_weight = F.softmax(alignment_score, dim=1)

        context_vector = torch.bmm(attention_weight, encoder_outputs)

        output = torch.cat((embedding, context_vector[0]), 1)
        output = F.log_softmax(self.classifier(output[0]), dim=1)

        return output, hidden, attention_weight

class Seq2Seq(BaseModel):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder
        self.decoder = BahdanauDecoder

    def forward(self, inputs):
        return inputs
