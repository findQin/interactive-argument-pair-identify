import torch

from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence


class RnnForSentencePairClassification(nn.Module):
    """
    -----TextCNN-----
    """
    def __init__(self, config):

        super().__init__()

        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)

        #region embedding 类似于TextCNN中的卷积操作
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size), stride=1)

        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom 上下各添加1个0

        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom 下添加一个0

        self.relu = nn.ReLU()

        self.linear = nn.Linear(config.num_filters*2, config.num_classes)

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
        # embed: (batch_size, max_seq_len, hidden_size)
        s1_embed = self.embedding(s1_ids)
        s2_embed = self.embedding(s2_ids)

        # s1_in: (batch_size, 1, max_seq_len, hidden_size)
        s1_in = s1_embed.unsqueeze(1)
        s2_in = s2_embed.unsqueeze(1)

        # (batch_size, num_filters, seq_len-3+1, 1)
        s1_in = self.conv_region(s1_in)  
        s2_in = self.conv_region(s2_in)

        #先卷积 再填充 等价于等长卷积  序列长度不变
        # [batch_size, num_filters, seq_len, 1]
        s1_in = self.padding1(s1_in)  
        s2_in = self.padding1(s2_in) 

        s1_in = self.relu(s1_in)
        s2_in = self.relu(s2_in)

        # [batch_size, num_filters, seq_len-3+1, 1]
        s1_in = self.conv(s1_in) 
        s2_in = self.conv(s2_in) 

        # [batch_size, num_filters, seq_len, 1]
        s1_in = self.padding1(s1_in)  
        s2_in = self.padding1(s2_in) 

        s1_in = self.relu(s1_in) 
        s2_in = self.relu(s2_in) 

        # [batch_size, num_filters, seq_len-3+1, 1]
        s1_in = self.conv(s1_in)  
        s2_in = self.conv(s2_in)

        # s1_in.size()[2]: seq_len-3+1
        while s1_in.size()[2] > 2: 
            s1_in = self._block(s1_in)  

        while s2_in.size()[2] > 2: 
            s2_in = self._block(s2_in) 

        # [batch_size, num_filters]
        s1_in = s1_in.squeeze()  
        s2_in = s2_in.squeeze()  

        hidden = torch.cat([s1_in, s2_in], dim=-1)

        hidden = self.linear(hidden).view(-1, self.num_classes)

        hidden = self.dropout(hidden)

        logits = nn.functional.softmax(hidden, dim=-1)
        
        # logits: (batch_size, num_classes)
        return logits


    def _block(self, x):

        #[batch_size, num_filters, seq_len-1, 1]
        x = self.padding2(x) 

        #长度减半
        #[batch_size, num_filters, （seq_len-1）/2, 1]
        px = self.max_pool(x) 
 
        #等长卷积 长度不变
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
 
        # 等长卷积 长度不变
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
 
        # Short Cut
        x = x + px
        return x