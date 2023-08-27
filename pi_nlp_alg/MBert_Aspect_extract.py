import torch.nn as nn


class MBert_Aspect_extract(nn.Module):
    def __init__(self, d_hid, d_inner_hid=None, dropout=0):
        super(MBert_Aspect_extract, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output
