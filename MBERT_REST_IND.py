import os
from os import chdir, getcwd
from pytorch_transformers import BertModel
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import math
import os, sys
import random
import numpy as np
from pi_nlp_alg.alg_bdparse import MBERT_DKnldg, MBERT_Parsefunc, mbert_datfeat, mbert_datfeatex
from pi_nlp_alg.mbert_alg import MBERT_ALG
import torch.nn.functional as FUN

project_dir =getcwd()
corpus_dir = os.path.join(project_dir, 'corpus')
res_dir = os.path.join(project_dir, 'MBERT_EVAL_Report')
alg_dir = os.path.join(project_dir, 'pi_nlp_alg')
tv_dir = os.path.join(project_dir, 'tv')
print(project_dir)


def mbert_mat_gen(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    mbert_dat_Val = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        mbert_dat_Val[:len(trunc)] = trunc
    else:
        mbert_dat_Val[-len(trunc):] = trunc
    return mbert_dat_Val

def mbert_data_knwldg_ft(datin_intp, datin_aspval, datin_concp, mbert_dbv):
    datin_intp = datin_intp.lower().strip()
    datin_concp = datin_concp.lower().strip()
    datin_aspval = datin_aspval.lower().strip()
    
    mbert_ind_dat_param1 = mbert_dbv.mbert_TextSent_gen(datin_intp + " " + datin_aspval + " " + datin_concp)            
    mbert_ind_dat_param2 = mbert_dbv.mbert_TextSent_gen(datin_aspval)
    mbert_ind_dat_param3 = np.sum(mbert_ind_dat_param2 != 0)
    mbert_ind_par1 = mbert_dbv.mbert_TextSent_gen('[CLS] ' + datin_intp + " " + datin_aspval + " " + datin_concp + ' [SEP] ' + datin_aspval + " [SEP]")
    mbert_ind_par3 = mbert_dbv.mbert_TextSent_gen(
        "[CLS] " + datin_intp + " " + datin_aspval + " " + datin_concp + " [SEP]")
    mbert_ind_par2 = np.asarray([0] * (np.sum(mbert_ind_dat_param1 != 0) + 2) + [1] * (mbert_ind_dat_param3 + 1))
    mbert_ind_par2 = mbert_mat_gen(mbert_ind_par2, mbert_dbv.max_seq_len)
    mbert_ind_par4 = mbert_dbv.mbert_TextSent_gen("[CLS] " + datin_aspval + " [SEP]")

    return mbert_ind_par1, mbert_ind_par2, mbert_ind_par3, mbert_ind_par4


if __name__ == '__main__':





    parser = argparse.ArgumentParser()

    algparam = parser.parse_args()
    algparam.bse_dir = os.getcwd();
    algparam.model_name = 'ps'
    algparam.dataset = 'c1'
    algparam.use_single_bert = False
    algparam.optimizer = 'adam'
    algparam.initializer = 'xavier_uniform_'
    algparam.learning_rate = 0.00001
    algparam.dropout = 0
    algparam.l2reg = 0.00001
    algparam.num_epoch = 1
    algparam.batch_size = 1
    algparam.log_step = 1
    algparam.logdir = 'log'
    algparam.embed_dim = 300
    algparam.hidden_dim = 300
    algparam.bert_dim = 768
    algparam.pretrained_bert_name ='bert-base-uncased'
    algparam.max_seq_len =80
    algparam.polarities_dim = 3
    algparam.hops = 3
    algparam.SRD = 3
    algparam.local_context_focus ='ps'
    algparam.seed = 2019
    algparam.device = 'cpu'

    model_class = {
        'ps': MBERT_ALG,
    }

    state_dict_paths = {
        'ps': algparam.bse_dir + '\\tv\\c1mdl.tv'
    }

    mbert_dbv = MBERT_DKnldg(algparam.max_seq_len, algparam.pretrained_bert_name)
    mbert_btmd_knw = BertModel.from_pretrained(algparam.pretrained_bert_name)
    model = model_class[algparam.model_name](mbert_btmd_knw, algparam).to(algparam.device)
    model.load_state_dict(torch.load(algparam.bse_dir + '\\tv\\c1mdl.tv', map_location=torch.device('cpu')))
    model.eval()
    torch.autograd.set_grad_enabled(False)

     

     # Negative Sentiment = CLASS = -1
     # INPUT: "the staff was so horrible to us ."
     #dat_in_init = 'the'
     #dat_in_aspct = 'staff'
     #dat_in_conc = 'was so horrible to us .'

     #mbert_ind_par1, mbert_ind_par2, mbert_ind_par3, mbert_ind_par4 = mbert_data_knwldg_ft(dat_in_init,dat_in_aspct,dat_in_conc,mbert_dbv)

     # Neutral Sentiment = CLASS 0
     #INPUT: Each table has a pot of boiling water sunken into its surface , and you get platters of thin sliced meats , various vegetables , and rice and glass noodles .
     #dat_in_init = 'Each table has a pot of boiling water sunken into its surface , and you get platters of thin sliced meats , various'
     #dat_in_aspct = 'vegetables'
     #dat_in_conc = ', and rice and glass noodles .'

     #mbert_ind_par1, mbert_ind_par2, mbert_ind_par3, mbert_ind_par4 = mbert_data_knwldg_ft(dat_in_init,dat_in_aspct,dat_in_conc,mbert_dbv)

    # POSITIVE SENTIMENT = CLASS 1
    #INPUT: This place is always very crowded and popular .
    #dat_in_init = 'This'
    #dat_in_aspct = 'place'
    #dat_in_conc = 'is always very crowded and popular .'

     #mbert_ind_par1, mbert_ind_par2, mbert_ind_par3, mbert_ind_par4 = mbert_data_knwldg_ft(dat_in_init,dat_in_aspct,dat_in_conc,mbert_dbv)

    # NEGATIVE SENTIMENT = CLASS -1 (NATURAL NOT IN DATABASE)
    #INPUT: The staff is too smart.
    #dat_in_init = 'The'
    #dat_in_aspct = 'staff'
    #dat_in_conc = 'is too smart'

    # NEGATIVE SENTIMENT = CLASS -1 (NATURAL NOT IN DATABASE)
    #INPUT: The staff is too smart.
    dat_in_init = 'The'
    dat_in_aspct = 'staff'
    dat_in_conc = 'is too smart'

    mbert_ind_par1, mbert_ind_par2, mbert_ind_par3, mbert_ind_par4 = mbert_data_knwldg_ft(dat_in_init,dat_in_aspct,dat_in_conc,mbert_dbv)


    mbert_ind_par1 = torch.tensor([mbert_ind_par1], dtype=torch.int64).to(algparam.device)
    mbert_ind_par2 = torch.tensor([mbert_ind_par2], dtype=torch.int64).to(algparam.device)
    mbert_ind_par3 = torch.tensor([mbert_ind_par3], dtype=torch.int64).to(algparam.device)
    mbert_ind_par4 = torch.tensor([mbert_ind_par4], dtype=torch.int64).to(algparam.device)

    nlpin_indat_tnsr = [mbert_ind_par1, mbert_ind_par2, mbert_ind_par3, mbert_ind_par4]
    nlpin_resdat_tnsr = model(nlpin_indat_tnsr)
    nlpin_resprobdat = FUN.softmax(nlpin_resdat_tnsr, dim=-1).cpu().numpy()
    print('ANALYZING THE FOLLOWING REVIEW FOR CORPUS 1:')
    print(dat_in_init,dat_in_aspct,dat_in_conc)
    print('PROBABLITIES EVALUATED = ', nlpin_resprobdat)
    print('SENTIMENT CLASSIFICATION = ', nlpin_resprobdat.argmax(axis=-1) - 1)


