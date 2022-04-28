import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextClassifier(nn.Module):
    def __init__(self, hidden_layers, hidden_size, dropout, TEXT, LABEL, pretrained=False):
        super(TextClassifier, self).__init__()
        self.hidden_size = hidden_size

        if pretrained:
            self.embeddings = nn.Embedding.from_pretrained(TEXT.vocab.vectors)
        else:
            self.embeddings = nn.Embedding(*TEXT.vocab.vectors.size())
            self.init_weights(self.embeddings.embedding_dim)

        self.lstm = nn.LSTM(input_size=self.embeddings.embedding_dim, dropout=dropout if hidden_size > 1 else 0,
            hidden_size=hidden_size, num_layers=hidden_layers, bidirectional=True, batch_first=True)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(LABEL.vocab.itos))
        )

    def init_weights(self, embedding_dim):
        irange = 0.5 / embedding_dim
        nn.init.uniform_(self.embeddings.weight.data, -irange, irange)

    def attention_net(self, lstm_output, final_hidden_state):
        attn_weights = self.attention(final_hidden_state)
        attn_weights = torch.bmm(attn_weights, nn.Tanh()(lstm_output).transpose(1, 2))
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.bmm(attn_weights, lstm_output).squeeze(1)
        return output, attn_weights

    def forward(self, x, length):
        embeds = self.embeddings(x)

        embeds_packed = pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted=False)
        output_packed, (final_h, _) = self.lstm(embeds_packed)
        output, _ = pad_packed_sequence(output_packed, batch_first=True)

        (forward_out, backward_out) = torch.chunk(output, 2, -1)
        output = forward_out + backward_out

        final_h = final_h.permute(1, 0, 2)
        final_h = torch.sum(final_h, dim=1).unsqueeze(1)

        output, attention = self.attention_net(output, final_h)
        output = self.linear(output)
        return output, attention
