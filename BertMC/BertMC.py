
from transformers.modeling_bert import BertModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class BertOrigin(nn.Module):
    """BERT model for multiple choice tasks. BERT + Linear

    Args:
        config: BertConfig 类对象， 以此创建模型
        num_choices: 选项数目，默认为 2.
    """

    def __init__(self, config, num_classes):
    
        super().__init__()
        
        self.num_choices = num_classes
        
        self.bert = BertModel.from_pretrained( config.bert_model_dir )
        
        for param in self.bert.parameters():
            param.requires_grad = True 
        
        self.dropout = nn.Dropout( config.hidden_dropout_prob )
        
        self.classifier = nn.Linear( config.hidden_size, 1 )
        

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Inputs:
            input_ids: [batch_size, num_choices, sequence_length]， 其中包含了词所对应的ids
            token_type_ids: 可选，[batch_size, num_choices, sequence_length]；0 表示属于句子 A， 1 表示属于句子 B
            attention_mask: 可选，[batch_size, num_choices, sequence_length]；区分 padding 与 token， 1表示是token，0 为padding
        """
        # flat_input_ids: [batch_size * num_choices, sequence_length]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        
        # flat_token_type_ids: [batch_size * num_choices, sequence_length]
        flat_token_type_ids = token_type_ids.view(
            -1, token_type_ids.size(-1))
            
        # flat_attention_mask: [batch_size * num_choices, sequence_length]
        flat_attention_mask = attention_mask.view(
            -1, attention_mask.size(-1))   

        _, pooled_output = self.bert(
            flat_input_ids, flat_token_type_ids, flat_attention_mask, encoder_hidden_states=False)
        
        # pooled_output: [batch_size * num_choices, 768]
        
        pooled_output = self.dropout( pooled_output )
        
        # logits: [batch_size * num_choices, 1]
        logits = self.classifier(pooled_output)

        # reshaped_logits: [batch_size, num_choices]
        reshaped_logits = logits.view(-1, self.num_choices)

        #dim=-1表示按行计算
        # reshaped_logits: (batch_size, num_choices)
        reshaped_logits = nn.functional.softmax(reshaped_logits, dim=-1)
        
        return reshaped_logits
