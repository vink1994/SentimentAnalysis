import torch
import torch.nn as nn
import copy
import numpy as np

from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention


class Mbert_impr_mech(nn.Module):
    def __init__(self, config, opt):
        super(Mbert_impr_mech, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class MBERT_ALG(nn.Module):
    def __init__(self, bert, opt):
        super(MBERT_ALG, self).__init__()
        self.bert_global_focus = bert
        self.bert_local_focus = copy.deepcopy(bert) if opt.use_single_bert else bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = Mbert_impr_mech(bert.config, opt)
        self.linear_double_cdm_or_cdw = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_triple_lcf_global = nn.Linear(opt.bert_dim * 3, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

   
    def Mbert_locfeat_extract(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len + 1, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i])-1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index)+asp_len/2
                                        - self.opt.SRD)/np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        bert_global_out, _ = self.bert_global_focus(text_bert_indices, bert_segments_ids)
        bert_local_out, _ = self.bert_local_focus(text_local_indices)

        masked_local_text_vec = self.Mbert_locfeat_extract(text_local_indices, aspect_indices)
        bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)
        out_cat = torch.cat((bert_local_out, bert_global_out), dim=-1)
        out_cat = self.linear_double_cdm_or_cdw(out_cat)

        self_attention_out = self.bert_SA(out_cat)
        self_attention_out = self.dropout(self_attention_out)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)
        return dense_out
