import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers.modeling_bert import BertModel


class BertForClassification(nn.Module):

    def __init__(self, config):
   
        super().__init__()

        self.bert = BertModel.from_pretrained( config.bert_model_path )

        for param in self.bert.parameters():
            param.requires_grad = True

        # text_cnn
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])

        #self.linear = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes

        # text_cnn
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

     # text_cnn
    def conv_and_pool(self, x, conv):
        # conv(x)处理后，输出的维度为[batch_size, num_filters, seq_length, 1]
        # x.squeeze(3)去掉最后一个维度 1
        # [batch_size, num_filters, seq_length]
        x = nn.functional.relu(conv(x)).squeeze(3)
        
        # max_pool1d处理，在最后一个维度上取一个最大值
        # 并通过x = x.squeeze(2)消除最后一个维度 seq_length
        # 最终输出维度为[batch_size, num_filters]
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        
        return x

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=False)

        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # bert_output[1]: (batch_size, hidden_size)

        #pooled_output = bert_output[1]

        #pooled_output = self.dropout(pooled_output)

        # (batch_size, sequence_length, hidden_size)
        encoder_out = bert_output[0]

        # 传统CNN做图像时的输入：[batch_size, in_channel, height, width]
        # unsqueeze(1)使数据在第二维增加一维，现在才能做卷积 [batch_size, 1, seq_len, embed_size]
        out = encoder_out.unsqueeze(1)

        # 通过不同大小的卷积核提取特征 并对池化结果进行横着拼接 [ batch_size, num_filters*len(filter_size) ]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)

        out = self.dropout(out)

        # (batch_size, num_classes)
        logits = self.fc_cnn(out)

        #logits = self.linear(pooled_output).view(batch_size, self.num_classes)

        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)

        return logits


