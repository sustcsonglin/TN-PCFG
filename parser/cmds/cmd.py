# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric,  UF1, LossMetric, UAS

import time

class CMD(object):
    def __call__(self, args):
        self.args = args

    def train(self, loader):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        train_arg = self.args.train
        for x, _ in t:

            self.optimizer.zero_grad()
            loss = self.model.loss(x)
            loss.backward()
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
            self.optimizer.step()

        return


    @torch.no_grad()
    def evaluate(self, loader, eval_dep=False, decode_type='mbr', model=None):
        if model == None:
            model = self.model
        model.eval()
        metric_f1 = UF1()
        if eval_dep:
            metric_uas = UAS()
        metric_ll = LikelihoodMetric()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        print('decoding mode:{}'.format(decode_type))
        print('evaluate_dep:{]'.format(eval_dep))
        for x, y in t:
            result = model.evaluate(x, decode_type=decode_type, eval_dep=eval_dep)
            metric_f1(result['prediction'], y['gold_tree'])
            metric_ll(result['partition'], x['seq_len'])
            if eval_dep:
                metric_uas(result['prediction_arc'], y['head'])
        if not eval_dep:
            return metric_f1, metric_ll
        else:
            return metric_f1, metric_uas, metric_ll




