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
import numpy, random
from pi_nlp_alg.alg_bdparse import MBERT_DKnldg, MBERT_Parsefunc, mbert_datfeat, mbert_datfeatex
from pi_nlp_alg.mbert_alg import MBERT_ALG
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import logging, sys

from time import strftime, localtime

project_dir =getcwd()
corpus_dir = os.path.join(project_dir, 'corpus')
res_dir = os.path.join(project_dir, 'MBERT_EVAL_Report')
alg_dir = os.path.join(project_dir, 'pi_nlp_alg')
tv_dir = os.path.join(project_dir, 'tv')
print(project_dir)
mbert_res_log = logging.getLogger()
mbert_res_log.setLevel(logging.INFO)
mbert_res_log.addHandler(logging.StreamHandler(sys.stdout))

time = '{}'.format(strftime("%y%m%d-%H%M%S", localtime()))
os.mkdir('MBERT_EVAL_Report\\{}'.format(time))
log_file = 'MBERT_EVAL_Report\\{}/{}.dat'.format(time, time)
mbert_res_log.addHandler(logging.FileHandler(log_file))


class MBERT_Initial_ALGO:
    def __init__(self, algparam):
        self.algparam = algparam

        mbert_dbv = MBERT_DKnldg(algparam.max_seq_len, algparam.pretrained_bert_name)
        mbert_btmd_knw = BertModel.from_pretrained(algparam.pretrained_bert_name)

        self.model = algparam.model_class(mbert_btmd_knw, algparam).to(algparam.device)



        mbert_corp_train = MBERT_Parsefunc(algparam.dataset_file['train'], mbert_dbv)
        mbert_corp_eval = MBERT_Parsefunc(algparam.dataset_file['test'], mbert_dbv)
        self.getmbert_corp_train = DataLoader(dataset=mbert_corp_train, batch_size=algparam.batch_size, shuffle=True)
        self.getmbert_corp_eval = DataLoader(dataset=mbert_corp_eval, batch_size=algparam.batch_size, shuffle=False)



    def _psinitengine(self):
        for child in self.model.children():
            if type(child) != BertModel:  
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.algparam.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    

    def mbert_param_eval(self, nlpobjectivefn, optimizer, lp_res_acc_evlcorpa=0):

        mbert_res_acc = 0
        res_m_f1_eval = 0
        alg_lp_cntrl = 0

        for alg_epoch_val in range(self.algparam.num_epoch):
            logging.info('>' * 100)
            logging.info('alg_epoch_val: {}'.format(alg_epoch_val))
            res_true, res_totcorpa = 0, 0
            for i_batch, sample_batched in enumerate(self.getmbert_corp_train):
                alg_lp_cntrl += 1


                self.model.load_state_dict(torch.load(self.algparam.bse_dir +'\\tv\\c1mdl.tv', map_location=torch.device('cpu')))
                self.model.eval()

                mbert_input_val = [sample_batched[col].to(self.algparam.device) for col in self.algparam.inputs_cols]

                mbert_op_val= self.model(mbert_input_val)
                mbert_exp_val = sample_batched['polarity'].to(self.algparam.device)
                mbert_err_val = nlpobjectivefn(mbert_op_val, mbert_exp_val)


                if alg_lp_cntrl % self.algparam.log_step == 0:
                    res_true += (torch.argmax(mbert_op_val, -1) == mbert_exp_val).sum().item()
                    res_totcorpa += len(mbert_op_val)
                    mbert_train_accuracy = res_true / res_totcorpa

                    mbert_train_acc, res_trnfscoe_evcorpa, mbert_training_report, res_trnconf_mat = self.MBERT_Aspect_trainingRes()
                    MBERT_EVAL_Accuracy, res_fscoe_evcorpa, res_report, res_conf_mat = self.mbert_aspect_res()
                    if MBERT_EVAL_Accuracy > mbert_res_acc:
                        mbert_res_acc = MBERT_EVAL_Accuracy

                        logging.info('M-BERT model training results')    
                        logging.info('PS ALGO ACCURACY:{}  ALGO f1 SCORE:{}'.format(round(mbert_train_acc * 100, 2),round(res_trnfscoe_evcorpa * 100, 2)))
                        logging.info('Classification Report:')
                        logging.info(mbert_training_report)
                        logging.info('Confusion Matrix:')
                        logging.info(res_trnconf_mat)

                        logging.info('M-BERT model evaluation results')   
                        logging.info('PS ALGO ACCURACY:{}  ALGO f1 SCORE:{}'.format(round(MBERT_EVAL_Accuracy * 100, 2),round(res_fscoe_evcorpa * 100, 2)))
                        logging.info('Classification Report:')
                        logging.info(res_report)
                        logging.info('Confusion Matrix:')
                        logging.info(res_conf_mat)
                    if res_fscoe_evcorpa > res_m_f1_eval:
                        res_m_f1_eval = res_fscoe_evcorpa
                    logging.info('Training Error: {:.4f}, Training Accuracy: {:.2f}, Evaluation Accuracy: {:.2f}, Evaluation F1 Score: {:.2f}'.format(mbert_err_val.item(),mbert_train_accuracy*100,MBERT_EVAL_Accuracy*100, res_fscoe_evcorpa*100))
        return mbert_res_acc, res_m_f1_eval
   
    def MBERT_Aspect_trainingRes(self):
        
        self.model.eval()
        var_eval_true, var_eval_totcorpa = 0, 0
        mat_eval_truevals, mat_eval_predvals = None, None
        with torch.no_grad():
            for v_bv, v_s_bv in enumerate(self.getmbert_corp_train):
                indat_nlp = [v_s_bv[col].to(self.algparam.device) for col in self.algparam.inputs_cols]
                trusenti_nlp = v_s_bv['polarity'].to(self.algparam.device)
                predsenti_nlp = self.model(indat_nlp)

                var_eval_true += (torch.argmax(predsenti_nlp, -1) == trusenti_nlp).sum().item()
                var_eval_totcorpa += len(predsenti_nlp)

                if mat_eval_truevals is None:
                    mat_eval_truevals = trusenti_nlp
                    mat_eval_predvals = predsenti_nlp
                else:
                    mat_eval_truevals = torch.cat((mat_eval_truevals, trusenti_nlp), dim=0)
                    mat_eval_predvals = torch.cat((mat_eval_predvals, predsenti_nlp), dim=0)

        mbert_train_acc = var_eval_true / var_eval_totcorpa

        mbert_training_report = classification_report(mat_eval_truevals.cpu(), torch.argmax(mat_eval_predvals, -1).cpu(), labels=[0, 1, 2], digits=4)
        res_trnconf_mat = confusion_matrix(mat_eval_truevals.cpu(), torch.argmax(mat_eval_predvals, -1).cpu(), labels=[0, 1, 2])
        res_trnfscoe_evcorpa = metrics.f1_score(mat_eval_truevals.cpu(), torch.argmax(mat_eval_predvals, -1).cpu(), labels=[0, 1, 2], average='macro')
        
        return mbert_train_acc, res_trnfscoe_evcorpa, mbert_training_report, res_trnconf_mat

    def mbert_aspect_res(self):
       
        self.model.eval()
        var_eval_true, var_eval_totcorpa = 0, 0
        mat_eval_truevals, mat_eval_predvals = None, None
        with torch.no_grad():
            for v_bv, v_s_bv in enumerate(self.getmbert_corp_eval):
                indat_nlp = [v_s_bv[col].to(self.algparam.device) for col in self.algparam.inputs_cols]
                trusenti_nlp = v_s_bv['polarity'].to(self.algparam.device)
                predsenti_nlp = self.model(indat_nlp)

                var_eval_true += (torch.argmax(predsenti_nlp, -1) == trusenti_nlp).sum().item()
                var_eval_totcorpa += len(predsenti_nlp)

                if mat_eval_truevals is None:
                    mat_eval_truevals = trusenti_nlp
                    mat_eval_predvals = predsenti_nlp
                else:
                    mat_eval_truevals = torch.cat((mat_eval_truevals, trusenti_nlp), dim=0)
                    mat_eval_predvals = torch.cat((mat_eval_predvals, predsenti_nlp), dim=0)

        MBERT_EVAL_Accuracy = var_eval_true / var_eval_totcorpa

        res_report = classification_report(mat_eval_truevals.cpu(), torch.argmax(mat_eval_predvals, -1).cpu(), labels=[0, 1, 2], digits=4)
        res_conf_mat = confusion_matrix(mat_eval_truevals.cpu(), torch.argmax(mat_eval_predvals, -1).cpu(), labels=[0, 1, 2])
        res_fscoe_evcorpa = metrics.f1_score(mat_eval_truevals.cpu(), torch.argmax(mat_eval_predvals, -1).cpu(), labels=[0, 1, 2], average='macro')
        
        return MBERT_EVAL_Accuracy, res_fscoe_evcorpa, res_report, res_conf_mat

    def mbert_top_func(self, repeats=1):

        
        nlpobjectivefn = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.algparam.optimizer(_params, lr=self.algparam.learning_rate, weight_decay=self.algparam.l2reg)

        lp_res_acc_evlcorpa = 0
        lp_res_fscre_evlcorpa = 0
        for i in range(repeats):
            
            logging.info('MBERT Model Execution on Restaurant Dataset')
            self._psinitengine()


            mbert_res_acc, res_m_f1_eval = self.mbert_param_eval(nlpobjectivefn, optimizer, lp_res_acc_evlcorpa=lp_res_acc_evlcorpa)

            lp_res_acc_evlcorpa = max(res_m_acc_eval, lp_res_acc_evlcorpa)
            lp_res_fscre_evlcorpa = max(res_m_f1_eval, lp_res_fscre_evlcorpa)
            logging.info('#' * 100)

        logging.info("lp_res_acc_evlcorpa:{}".format(lp_res_acc_evlcorpa * 100))
        logging.info("lp_res_fscre_evlcorpa:{}".format(lp_res_fscre_evlcorpa * 100))
        return lp_res_acc_evlcorpa * 100, lp_res_fscre_evlcorpa * 100


def mbert_DS_top_func():

    parser = argparse.ArgumentParser()
 
    algparam = parser.parse_args()

    algparam.bse_dir = os.getcwd()
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

    if algparam.seed is not None:
        random.seed(algparam.seed)
        numpy.random.seed(algparam.seed)
        torch.manual_seed(algparam.seed)
        torch.cuda.manual_seed(algparam.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'ps': MBERT_ALG,
    }

    dataset_files = {
       'c1': {
            'train': algparam.bse_dir + '\\corpus\\1\\C1_Train.rdat',
            'test': algparam.bse_dir + '\\corpus\\1\\C1_Test.rdat'
            }

    }

    input_colses = {
        'ps': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
        
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        
    }
    optimizers = {
        'adam': torch.optim.Adam,  
    }
    algparam.model_class = model_classes[algparam.model_name]
    algparam.dataset_file = dataset_files[algparam.dataset]
    algparam.inputs_cols = input_colses[algparam.model_name]
    algparam.initializer = initializers[algparam.initializer]
    algparam.optimizer = optimizers[algparam.optimizer]



    nlp_ps_engine = MBERT_Initial_ALGO(algparam)
    return nlp_ps_engine.mbert_top_func()  


if __name__ == '__main__':

    mbert_DS_top_func()
