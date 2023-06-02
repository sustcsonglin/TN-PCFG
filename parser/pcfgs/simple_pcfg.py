from turtle import pd
from typing import final
from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import  stripe, diagonal_copy_, checkpoint, diagonal, stripe_add_
import torch
from parser.triton.fn import _merge, _log_then_diagonal_copy_



import pdb


class SimplePCFG_Triton(PCFG_base):
    def __init__(self):
        super(SimplePCFG_Triton, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)


    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None):
        assert viterbi is not True
        # B, L, r_p
        unary = rules['unary'].clone()
        # B, L, r_m
        root = rules['root'].exp()        

        # r_m, r_m 
        L = rules['left_m']
        R = rules['right_m']
        # r_p, r_p
        L_p = rules['left_p']
        R_p = rules['right_p']
        LR = torch.cat([L, R], dim=-1)

        r_p = unary.shape[-1]
        r_m = L.shape[1]        
    
        batch, N, *_ = unary.shape
        N += 1
        # for estimating marginals.
        if s_span is None:
            span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()        
        unary = torch.einsum('bnp, pq -> bnq',  unary ,torch.cat([L_p, R_p], dim=-1))

        alpha_c = unary.new_zeros(batch, N, N,  2, r_m)
        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)
        
        # w: span width
        for w in range(2, N):
            n = N - w      
            normalizer = alpha_c.new_zeros(batch, n)            
            out, normalizer = _merge(normalizer, diagonal(span_indicator, w), alpha_c)
            if w < N-1:                                
                out = torch.einsum('blr, rq -> blq', out, LR)                
                alpha_c = _log_then_diagonal_copy_(out, normalizer, alpha_c)

        logZ = (torch.einsum('bnr, br -> b', out, root) + 1e-9).log() + normalizer.squeeze(1)

        if not mbr and not viterbi:
            return {'partition': logZ}

        elif marginal:
            logZ.sum().backward()                        
            return {'marginal': span_indicator.grad}
        
        else:
            return {                
                "prediction": self._get_prediction(logZ, span_indicator, lens, mbr=True),
                "partition": logZ
            }



class SimplePCFG_Triton_Batch(PCFG_base):
    def __init__(self):
        super(SimplePCFG_Triton_Batch, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)


    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None):
        assert viterbi is not True
        # B, L, r_p
        unary = rules['unary'].clone()
        # B, L, r_m
        root = rules['root'].exp()

        # r_m, r_m 
        L = rules['left_m']
        R = rules['right_m']
        # r_p, r_p
        L_p = rules['left_p']
        R_p = rules['right_p']
        LR = torch.cat([L, R], dim=-1)
                 
        r_p = unary.shape[-1]
        r_m = L.shape[-2]        

        batch, N, *_ = unary.shape
        N += 1
        # for estimating marginals.
        if s_span is None:
            span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()        


        unary = torch.einsum('bnp, bpq -> bnq',  unary ,torch.cat([L_p, R_p], dim=-1))

        alpha_c = unary.new_zeros(batch, N, N,  2, r_m)

        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)
        
        # w: span width
        for w in range(2, N):
            n = N - w      
            normalizer = alpha_c.new_zeros(batch, n)
            
            out, normalizer = _merge(normalizer, diagonal(span_indicator, w), alpha_c)

            if w < N-1:                                
                out = torch.einsum('blr, brq -> blq', out, LR)                
                alpha_c = _log_then_diagonal_copy_(out, normalizer, alpha_c)
        
        logZ = (torch.einsum('bnr, br -> b', out, root) + 1e-9).log() + normalizer.squeeze(1)

        if not mbr and not viterbi:
            return {'partition': logZ}

        elif marginal:
            logZ.sum().backward()
            
            return {'marginal': span_indicator.grad}

        else:
            return {
                
                "prediction": self._get_prediction(logZ, span_indicator, lens, mbr=True),
                "partition": logZ
            }



