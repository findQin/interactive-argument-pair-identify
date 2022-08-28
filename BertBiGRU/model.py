import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers.modeling_bert import BertModel


class BertForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()

        self.bert = BertModel.from_pretrained(config.bert_model_path)

        for param in self.bert.parameters():
            param.requires_grad = True

        embedding_dim = self.bert.config.to_dict()['hidden_size']

        # -----定义GRU-----
        self.rnn = nn.GRU(embedding_dim,
                          config.rnn_hidden_dim,
                          num_layers = 2,
                          bidirectional = True,
                          batch_first = True,
                          dropout = 0.25)


        self.linear = nn.Linear(config.rnn_hidden_dim * 2 , config.num_classes)

        self.dropout = nn.Dropout(config.dropout)

        self.num_classes = config.num_classes


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
        encoder_output = bert_output[0]

        # ----GRU-----
        self.rnn.flatten_parameters()
        
        _ , hidden = self.rnn( encoder_output )

        #hidden = [n_layers * n_dirs, batch_size, rnn_hidden_dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        #hidden = [batch_size, rnn_hidden_dim*2]

        logits = self.linear(hidden).view(batch_size, self.num_classes)

        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)

        return logits



