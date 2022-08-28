import torch

from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence


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

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])

        self.rnn = nn.GRU(
            config.hidden_size, hidden_size=config.hidden_size, 
            bidirectional=False, batch_first=True)

        self.linear = nn.Linear(config.num_filters*len(config.filter_sizes)*2, config.num_classes)

        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes


    def conv_and_pool(self, x, conv):
        # conv(x)处理后，输出的维度为[batch_size, num_filters, some_seq_len, 1]
        # x.squeeze(3)去掉最后一个维度 1
        # [batch_size, num_filters, some_seq_len]
        x = F.relu(conv(x)).squeeze(3)
        
        # max_pool1d处理，在最后一个维度上取一个最大值[batch_size, num_filters, 1]
        # 并通过x = x.squeeze(2)消除最后一个维度 seq_length
        # 最终输出维度为[batch_size, num_filters]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        return x


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
        # embed: (batch_size, max_seq_len, hidden_size)
        s1_embed = self.embedding(s1_ids)
        s2_embed = self.embedding(s2_ids)

        # s1_in: (batch_size, 1, max_seq_len, hidden_size)
        s1_in = s1_embed.unsqueeze(1)
        s2_in = s2_embed.unsqueeze(1)

        #  [ batch_size, num_filters*len(filter_size) ]
        s1_out = torch.cat([self.conv_and_pool(s1_in, conv) for conv in self.convs], 1)
        s2_out = torch.cat([self.conv_and_pool(s2_in, conv) for conv in self.convs], 1)

        # embed: (batch_size, max_seq_len, hidden_size)
        # s1_packed: PackedSequence = pack_padded_sequence(
        #     s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)

        # s2_packed: PackedSequence = pack_padded_sequence(
        #     s2_embed, s2_lengths, batch_first=True, enforce_sorted=False)

        # packed: (sum(lengths), hidden_size)
        #self.rnn.flatten_parameters()

        # s1_packed是一个PackedSequence对象, 可直接送入rnn中
        # gru有两个输出，output和hidden
        # output：(batch_size, seq_len,  num_dir * hidden_size)GRU最后一层的特征输出h_t
        # hidden: (num_layers * num_dirs, batch, hidden_size)代表t=seq_len最后一个神经元的隐藏状态
        # s1_output, _ = self.rnn(s1_packed)
        # s2_output, _ = self.rnn(s2_packed)

        # 此时的s1_output也是一个PackedSequence对象
        # 所以需要用pad_packed_sequence还原回本来的样子
        # s1_output, _ = pad_packed_sequence(s1_output, batch_first=True, total_length=s1_embed.size(1))
        # s2_output, _ = pad_packed_sequence(s2_output, batch_first=True, total_length=s2_embed.size(1))

        # s1_out: (batch_size, 1, seq_len,  num_dir * hidden_size)
        #s1_out = s1_output.unsqueeze(1)
        #s2_out = s2_output.unsqueeze(1)

        hidden = torch.cat([s1_out, s2_out], dim=-1)

        hidden = self.linear(hidden).view(-1, self.num_classes)

        hidden = self.dropout(hidden)

        logits = nn.functional.softmax(hidden, dim=-1)
        
        # logits: (batch_size, num_classes)
        return logits