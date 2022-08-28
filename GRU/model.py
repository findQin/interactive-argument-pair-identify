import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers.modeling_bert import BertModel


class RnnForSentencePairClassification(nn.Module):
    """Unidirectional GRU model for sentences pair classification.
    2 sentences use the same encoder and concat to a linear model.
    """
    def __init__(self, config):
        """Initialize the model with config dict.
        Args:
            config: python dict must contains the attributes below:
                config.vocab_size: vocab size
                config.hidden_size: RNN hidden size and embedding dim
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """

        super().__init__()

        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)

        self.rnn = nn.GRU(
            config.hidden_size, hidden_size=config.hidden_size, 
            bidirectional=False, batch_first=True)

        self.linear = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.linear1 = nn.Linear(config.hidden_size * 4, config.num_classes)
        self.linear2 = nn.Linear(config.hidden_size * 6, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes


    def forward(self, s1_ids, s2_ids, s1_lengths, s2_lengths):
        """Forward inputs and get logits.
        Args:
            s1_ids: (batch_size, max_seq_len)
            s2_ids: (batch_size, max_seq_len)
            s1_lengths: (batch_size)
            s2_lengths: (batch_size)
        Returns:
            logits: (batch_size, num_classes)
        """

        batch_size = s1_ids.shape[0]

        # ids: (batch_size, max_seq_len)
        s1_embed = self.embedding(s1_ids)
        s2_embed = self.embedding(s2_ids)

        # embed: (batch_size, max_seq_len, hidden_size)
        s1_packed: PackedSequence = pack_padded_sequence(
            s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)

        s2_packed: PackedSequence = pack_padded_sequence(
            s2_embed, s2_lengths, batch_first=True, enforce_sorted=False)

        # packed: (sum(lengths), hidden_size)
        self.rnn.flatten_parameters()

        # s1_hidden: (n_layers*n_dir, batch_size, hidden_size)
        _, s1_hidden = self.rnn(s1_packed)
        _, s2_hidden = self.rnn(s2_packed)

        # s1_hidden: (batch_size, hidden_size*2)
        #s1_hidden = torch.cat((s1_hidden[-2,:,:], s1_hidden[-1,:,:]), dim=1)
        #s2_hidden = torch.cat((s2_hidden[-2,:,:], s2_hidden[-1,:,:]), dim=1)

        # s1_hidden: (batch_size, n_layers*n_dir*hidden_size)
        s1_hidden = s1_hidden.view(batch_size, -1)
        s2_hidden = s2_hidden.view(batch_size, -1)

        hidden = torch.cat([s1_hidden, s2_hidden], dim=-1)

        hidden = self.linear(hidden).view(-1, self.num_classes)

        hidden = self.dropout(hidden)

        logits = nn.functional.softmax(hidden, dim=-1)
        
        # logits: (batch_size, num_classes)
        return logits