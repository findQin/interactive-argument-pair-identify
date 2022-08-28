import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from copy import deepcopy


class TransformerForSentencePairClassification(nn.Module):
    """
    -----TextCNN-----
    """
    def __init__(self, config):

        super().__init__()

        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=0)

        #位置编码
        self.postion_embedding = Positional_Encoding(config.embed_size, config.pad_size, config.dropout, config.device)

        #transformer encoder block
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        #多个transformer encoder block
        self.encoders = nn.ModuleList([
            deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        #输出层
        self.fc1 = nn.Linear(config.max_seq_len * config.dim_model*2, config.num_classes)

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

        # 添加位置编码 (batch, max_seq_len, hidden_size)
        s1_out = self.postion_embedding(s1_embed)
        s2_out = self.postion_embedding(s2_embed)

        for encoder in self.encoders: #通过多个ender block
            s1_out = encoder(s1_out)  #(batch, seq_len, dim_model)
            s2_out = encoder(s2_out)

        s1_out = s1_out.view(s1_out.size(0), -1) #（batch, seq_len*dim_model）
        s2_out = s2_out.view(s2_out.size(0), -1)

        hidden = torch.cat([s1_out, s2_out], dim=-1)

        hidden = self.linear(hidden).view(-1, self.num_classes)

        hidden = self.dropout(hidden)

        logits = nn.functional.softmax(hidden, dim=-1)
        
        # logits: (batch_size, num_classes)
        return logits


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super().__init__()
        #多头注意力机制
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        #两个全连接层
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)
 
    def forward(self, x): #x (batch, seq_len, embed_size)  embed_size = dim_model
        out = self.attention(x) #计算多头注意力结果 (batch, seq_len, dim_model)
        out = self.feed_forward(out) #通过两个全连接层增加 非线性转换能力 (batch, seq_len, dim_model)
        return out


class Positional_Encoding(nn.Module):
    #位置编码
    def __init__(self, embed, pad_size, dropout, device):
        super().__init__()
        self.device = device

        #利用sin cos生成绝对位置编码
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x): 
        #token embedding + 绝对位置编码
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        #再通过dropout
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super().__init__()
 
    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        #Q与K的第2、3维转置计算内积  (batch*num_head, seq_len, seq_len)
        attention = torch.matmul(Q, K.permute(0, 2, 1))

        if scale: #作缩放 减小结果的方差 
            attention = attention * scale

        attention = F.softmax(attention, dim=-1) #转换为权重
        context = torch.matmul(attention, V) #再与V运算 得到结果 (batch*num_head,seq_len,dim_head)

        return context


class Multi_Head_Attention(nn.Module):
    #多头注意力机制 encoder block的第一部分
    def __init__(self, dim_model, num_head, dropout=0.0):
        super().__init__()

        self.num_head = num_head  #头数 
        assert dim_model % num_head == 0 #必须整除
        self.dim_head = dim_model // self.num_head 

        #分别通过三个Dense层 生成Q、K、V
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)

        #Attention计算
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)

        #层归一化
        self.layer_norm = nn.LayerNorm(dim_model)

 
    def forward(self, x): #（batch, seq_len, dim_model）
        batch_size = x.size(0) 

        # Q,K,V维度 (batch, seq_len, dim_head*num_head)
        Q = self.fc_Q(x)  
        K = self.fc_K(x)
        V = self.fc_V(x)

        #沿第三个维度进行切分 切分为num_head份 再沿第一个维度拼接 多个注意力头并行计算
        #Q,K,V维度 (batch*num_head, seq_len, dim_head)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head) 
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        # 缩放因子 dim_head的开放取倒数 对内积结果进行缩放 减小结果的方差 有利于训练
        scale = K.size(-1) ** -0.5  

        #attention计算 多个注意力头并行计算（矩阵运算）
        context = self.attention(Q, K, V, scale)
        
        #多头注意力计算结果 沿第一个维度进行切分 再沿第三个维度拼接 
        # 转为原来的维度(batch, seq_len, dim_head*num_head)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        
        #(batch, seq_len, dim_model)
        out = self.fc(context)
        out = self.dropout(out)

        # 残差连接
        out = out + x  
        out = self.layer_norm(out)

        return out


class Position_wise_Feed_Forward(nn.Module):
    #encoder block的第二部分
    def __init__(self, dim_model, hidden, dropout=0.0):
        #定义两个全连接层 多头注意力的计算结果 通过两个全连接层 增加非线性 
        super().__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
 
    def forward(self, x): # (batch, seq_len, dim_model)
        out = self.fc1(x)  # (batch, seq_len, hidden)

        out = F.relu(out)
        out = self.fc2(out) # (batch, seq_len, dim_model)

        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out) #层归一化

        return out