import torch
from torch import nn
from transformers.modeling_bert import BertModel, BertPooler
from torch.autograd import Variable



def seperate_seq(sequence_output, sc_len, bc_len):

    sc_seq_output = sequence_output.new(sequence_output.size()).zero_()

    bc_seq_output = sequence_output.new(sequence_output.size()).zero_()

    # sc_len: (batch_size)
    for i in range( sc_len.size(0) ):

        sc_seq_output[i, :sc_len[i]] = sequence_output[i, 1 : sc_len[i] + 1]
                
        bc_seq_output[i, :bc_len[i]] = sequence_output[i, sc_len[i] + 2 : sc_len[i] + bc_len[i] + 2]

    return sc_seq_output, bc_seq_output



def masked_softmax(vector, seq_lens):

    mask = vector.new(vector.size()).zero_()

    for i in range( seq_lens.size(0) ):
        mask[i, :, :seq_lens[i]] = 1

    mask = Variable(mask, requires_grad=False)

    # mask = None
    if mask is None:
        result = nn.functional.softmax(vector, dim=-1)
    else:
        # (vector * mask) 对应项相乘,
        # 对应 mask 中为 1 的在 vector 中保留, mask中为 0 的对应 vector 中为 0
        result = nn.functional.softmax(vector * mask, dim=-1)

        result = result * mask

        result = result / ( result.sum(dim=-1, keepdim=True) + 1e-13 )

    return result



class FuseNet(nn.Module):
    def __init__(self, config):
        super(FuseNet, self).__init__()

        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(2*config.hidden_size, 2*config.hidden_size)

    def forward(self, inputs):
        p, q = inputs

        lq = self.linear(q)

        lp = self.linear(p)

        mid = nn.Sigmoid()(lq+lp)

        output = p * mid + q * (1 - mid)

        return output



class SSingleMatchNet(nn.Module):
    def __init__(self, config):
        super(SSingleMatchNet, self).__init__()

        self.map_linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)

        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.drop_module = nn.Dropout(2 * config.dropout)

        self.rank_module = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, inputs):
        
        # proj_p: (batch_size, seq_len, hidden_size) (8, 512, 768)
        proj_p, proj_q, seq_len = inputs

        # q = W*q + b
        trans_q = self.trans_linear( proj_q )

        # G = p * q^T
        # G is attention weight between p and q
        att_weights = proj_p.bmm( torch.transpose(trans_q, 1, 2) )

        # G = masked_softmax( G )
        att_norm = masked_softmax(att_weights, seq_len)

        #  E = G * q (E represents q-aware p representation)
        att_vec = att_norm.bmm( proj_q )

        # RELU( E * W )
        # output = nn.ReLU()( self.trans_linear( att_vec ) )

        return att_vec



class BertForClassification(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.bert = BertModel.from_pretrained( config.bert_model_path )

        for param in self.bert.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout( config.dropout )

        self.num_classes = config.num_classes

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.classifier2 = nn.Linear(2*config.hidden_size, config.num_classes)

        self.classifier3 = nn.Linear(3*config.hidden_size, config.num_classes)

        self.classifier4 = nn.Linear(4*config.hidden_size, 1)

        self.classifier6 = nn.Linear(6*config.hidden_size, 1)

        self.ssmatch = SSingleMatchNet(config)


    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, sc_len=None, bc_len=None):
        """Forward inputs and get logits.
        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)
        Returns:
            logits: (batch_size, num_classes)
        """

        # sequence_output: (batch_size, max_seq, hidden_size)
        # pooled_output: (batch_size, hidden_size)
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, encoder_hidden_states=False)


        sc_seq_output, bc_seq_output = seperate_seq(sequence_output, sc_len, bc_len)
       
        # bc-aware sc representation
        sc_aware_bc_output = self.ssmatch([sc_seq_output, bc_seq_output, bc_len+1])

        cat_pool = torch.cat([sc_aware_bc_output, bc_seq_output], 2)

        cat_pooled, _ = cat_pool.max(1)

        # output_pool = self.dropout( cat_pooled )

        match_logits = self.classifier2( cat_pooled )

        match_reshaped_logits = match_logits.view(-1, self.num_classes) 

        logits = nn.functional.softmax(match_reshaped_logits, dim=-1)

        return logits

