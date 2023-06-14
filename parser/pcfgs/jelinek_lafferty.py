from turtle import pd
from typing import final
from uu import Error
from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import  stripe, diagonal_copy_, checkpoint, diagonal, stripe_add_
import torch
from parser.triton.fn import _merge, _log_then_diagonal_copy_, _merge_w_split


import pdb
    


class Jelinek_Lafferty(PCFG_base):
    def __init__(self):
        super(Jelinek_Lafferty, self).__init__()

    def loss(self, rules, lens):
        return self._inside2(rules, lens)

    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None, chunk=None):
        assert viterbi is not True
        # B, L, r_p
        unary = rules['unary']
    
        # B, L, r_m
        root = rules['root']

        # r_m, r_m 
        L = rules['left_m'].contiguous()
        R = rules['right_m'].contiguous()
        P = rules['P'].contiguous()

        split = rules['split']
        # r_p, r_p
        # L_p = L 
        # R_p = R 
        # LR = torch.cat([L, R], dim=-1)

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

        unary = unary + split[:, 0][None, None, :]
        split = split[:, 1].contiguous()

        unary_max = unary.max(-1)[0]
        
        unary = (unary - unary_max.unsqueeze(-1)).exp()        
        left_s = unary.new_zeros(batch, N, N, r_m)
        right_s = unary.new_zeros(batch, N, N, r_m)

        # pi = unary.new_zeros(batch, N, N, r_m)
        # pi_need_parent = unary.new_zeros(batch, N, N, r_m)

        diagonal_copy_(
            left_s, 
             (torch.einsum('bnp, pq -> bnq',  unary , L) + 1e-9).log() + unary_max.unsqueeze(-1),
             1
        )

        diagonal_copy_(
            right_s,
            (torch.einsum('bnp, pq -> bnq', unary, R) + 1e-9).log() + unary_max.unsqueeze(-1),
            1
        )

        # unary = unary @ P 

        # diagonal_copy_(
        #     pi, 
        #     (unary + 1e-9).log() + unary_max.unsqueeze(-1), 1
        # )
        
        # diagonal_copy_(
        #     pi_need_parent,
        #     ((unary @ R) + 1e-9).log() + unary_max.unsqueeze(-1), 1
        # )

        for w in range(2, N):
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            # W = stripe(pi_need_parent, n, (w-1), (1,w), 0)
            x = (Y + Z).logsumexp(-2) + span_indicator[:, torch.arange(n), w + torch.arange(n)].unsqueeze(-1) + split[None, None, :]
            
            if w + 1 < N:
                x_max = x.max(-1)[0]
                x = (x - x_max.unsqueeze(-1)).exp()                
                diagonal_copy_(left_s, (x @ L + 1e-9).log() + x_max.unsqueeze(-1) , w)
                diagonal_copy_(right_s, (x @ R + 1e-9).log() + x_max.unsqueeze(-1), w)
    
        logZ = (x.squeeze(1) + root).logsumexp(-1)
        
        if not mbr and not viterbi:
            return {'partition': logZ}

        elif marginal:
            logZ.sum().backward()                        
            return {'marginal': span_indicator.grad}
        
        else:
            return {                
                "prediction": self._get_prediction(logZ, span_indicator, lens, mbr=True, chunk=chunk),
                "partition": logZ
            }

    @torch.enable_grad()
    def _inside_triton(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None, chunk=None):
        assert viterbi is not True
        # B, L, r_p
        unary = rules['unary']
        # B, L, r_m
        root = rules['root'].exp()        

        # r_m, r_m 
        L = rules['left_m'].contiguous()
        R = rules['right_m'].contiguous()

        split = rules['split']
        # r_p, r_p
        # L_p = L 
        # R_p = R 
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

        unary = unary + split[:, 0][None, None, :]
        split = split[:, 1].contiguous()

        with torch.no_grad():
            unary_max = unary.max(-1)[0]
        

        unary = (unary - unary_max.unsqueeze(-1)).exp()        
        unary = torch.einsum('bnp, pq -> bnq',  unary , LR)

        alpha_c = unary.new_zeros(batch, N, N,  2, r_m)
        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)
        
        # w: span width
        for w in range(2, N):
            n = N - w      
            normalizer = alpha_c.new_zeros(batch, n)            
            out, normalizer = _merge_w_split(normalizer, diagonal(span_indicator, w), split, alpha_c)

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
                "prediction": self._get_prediction(logZ, span_indicator, lens, mbr=True, chunk=chunk),
                "partition": logZ
            }


    @torch.enable_grad()
    def conclusion_1999_next_preterminal(self, rules, lens):
        # B, L, r_p
        unary = rules['unary']       
        # B, L, r_m
        root = rules['root']

        # r_m, r_m 
        L = rules['left_m'].contiguous()
        R = rules['right_m'].contiguous()
        P = rules['P'].contiguous()

        split = rules['split']
        # r_p, r_p
        # L_p = L 
        # R_p = R 
        # LR = torch.cat([L, R], dim=-1)

        r_p = unary.shape[-1]
        r_m = L.shape[1]        
    
        batch, N, *_ = unary.shape
        N += 1
        
        # for estimating marginals.
        # if s_span is None:
        #     span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)
        # else:
        #     span_indicator = s_span
        #     if mbr or viterbi:
        #         span_indicator = span_indicator.detach().clone().requires_grad_(True)
        #     unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        split_unary = split[:, 0].contiguous()
        split = split[:, 1].contiguous()

        unary = unary + split_unary[None, None, :]

        unary_max = unary.max(-1)[0]
        
        unary = (unary - unary_max.unsqueeze(-1)).exp()        
        left_s = unary.new_zeros(batch, N, N, r_m)
        right_s = unary.new_zeros(batch, N, N, r_m)
    
        pi = unary.new_zeros(batch, N, N, r_m, r_m)
        pi_need_parent = unary.new_zeros(batch, N, N, r_m, r_m)

        diagonal_copy_(
            left_s, 
             (torch.einsum('bnp, pq -> bnq', unary , L) + 1e-9).log() + unary_max.unsqueeze(-1),
             1
        )

        diagonal_copy_(
            right_s,
            (torch.einsum('bnp, pq -> bnq', unary, R) + 1e-9).log() + unary_max.unsqueeze(-1),
            1
        )

        # unary = (P + 1e-9).log()

        diagonal_copy_(
            pi, 
            ((P+1e-9).log() + split_unary[:, None])[None, None, ...].expand(batch, N-1, r_m, r_m), 
            1
        )
        
        diagonal_copy_(
            pi_need_parent,
            (((P @ R) + 1e-9).log() + split_unary[:, None])[None, None, ...].expand(batch, N-1, r_m, r_m),
            1
        )

        for w in range(2, N):
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            W = stripe(pi_need_parent, n, (w-1), (1, w), 0)
            x = (Y + Z).logsumexp(-2) + split[None, None, :]            

            if w + 1 < N:
                x_max = x.max(-1)[0]
                x = (x - x_max.unsqueeze(-1)).exp()                
                diagonal_copy_(left_s, (x @ L + 1e-9).log() + x_max.unsqueeze(-1), w)
                diagonal_copy_(right_s, (x @ R + 1e-9).log() + x_max.unsqueeze(-1), w)
            
            x_pi = (Y.unsqueeze(-2) + W).logsumexp(-3) + split[None, None, None, :]
                        
            x_pi_max = x_pi.max(-1)[0]
            x_pi = (x_pi - x_pi_max.unsqueeze(-1)).exp()
            x_pi = x_pi @ P
            
            diagonal_copy_(
                pi, (x_pi + 1e-9).log() + x_pi_max.unsqueeze(-1), w
            )

            if w + 1 < N:
                x_pi_need_parent = x_pi @ R             
                diagonal_copy_(
                    pi_need_parent, (x_pi_need_parent + 1e-9).log() + x_pi_max.unsqueeze(-1), w
                )

        # (b, N-1)
        marginal = (pi[:, 0, 1:, :] + root.unsqueeze(1).unsqueeze(-2)).logsumexp(-1).softmax(-1)                

        return {'marginal': marginal}
        
    @torch.enable_grad()
    def marginal_next_preterminal(self, rules, lens):
        # B, L, r_p
        unary = rules['unary']       
        # B, L, r_m
        root = rules['root']

        # r_m, r_m 
        L = rules['left_m'].contiguous()
        R = rules['right_m'].contiguous()
        P = rules['P'].contiguous()
        split = rules['split']
        # r_p, r_p
        # L_p = L 
        # R_p = R 
        # LR = torch.cat([L, R], dim=-1)

        r_p = unary.shape[-1]
        r_m = L.shape[1]            
        batch, N, *_ = unary.shape
        N += 1        
        # for estimating marginals.
        indicator = unary.new_zeros(*unary.shape).requires_grad_(True)
        split_unary = split[:, 0].contiguous()
        split = split[:, 1].contiguous()
        unary = unary + split_unary[None, None, :]

        unary_max = unary.max(-1)[0]
        unary = (unary - unary_max.unsqueeze(-1)).exp()        

        left_s = unary.new_zeros(batch, N, N, r_m)
        right_s = unary.new_zeros(batch, N, N, r_m)
        pi_for_marginal = unary.new_zeros(batch, N, N, r_m)
        pi_for_marginal_need_parent = unary.new_zeros(batch, N, N, r_m)
                
        # pi = unary.new_zeros(batch, N, N, r_m)
        # pi_need_parent = unary.new_zeros(batch, N, N, r_m)

        diagonal_copy_(
            left_s, 
             (torch.einsum('bnp, pq -> bnq',  unary , L) + 1e-9).log() + unary_max.unsqueeze(-1),
             1
        )

        diagonal_copy_(
            right_s,
            (torch.einsum('bnp, pq -> bnq', unary, R) + 1e-9).log() + unary_max.unsqueeze(-1),
            1
        )

        # unary = unary
        # diagonal_copy_(
            # , 
            # (unary + 1e-9).log() + unary_max.unsqueeze(-1), 1
        #)

        tmp = ((split_unary.unsqueeze(1) + (P+1e-9).log())[None, None, ...] + indicator[:, :, :, None]).logsumexp(-2)
        
        diagonal_copy_(
            pi_for_marginal,
            tmp, 
            1
        )

        tmp_max = tmp.max(-1)[0]
        tmp = (tmp - tmp_max.unsqueeze(-1)).exp()        
        
        diagonal_copy_(
            pi_for_marginal_need_parent,
            ((tmp @ R) + 1e-9).log() + tmp_max.unsqueeze(-1),                         
            1
        )

        for w in range(2, N):
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            W = stripe(pi_for_marginal_need_parent, n, (w-1), (1,w), 0)
            
            x = (Y + Z).logsumexp(-2) + split[None, None, :]            
            
            if w + 1 < N:
                x_max = x.max(-1)[0]
                x = (x - x_max.unsqueeze(-1)).exp()                
                diagonal_copy_(left_s, (x @ L + 1e-9).log() + x_max.unsqueeze(-1) , w)
                diagonal_copy_(right_s, (x @ R + 1e-9).log() + x_max.unsqueeze(-1), w)
            
            x_pi = (Y + W).logsumexp(-2) + split[None, None, :]
            x_pi_max = x_pi.max(-1)[0]
            x_pi = (x_pi - x_pi_max.unsqueeze(-1)).exp()
            x_pi = x_pi @ P
            
            diagonal_copy_(
                pi_for_marginal, (x_pi + 1e-9).log() + x_pi_max.unsqueeze(-1), w
            )

            if w + 1 < N:
                x_pi_need_parent = x_pi @ R             
                diagonal_copy_(
                    pi_for_marginal_need_parent, (x_pi_need_parent + 1e-9).log() + x_pi_max.unsqueeze(-1), w
                )

        log_prefix_prob = (pi_for_marginal[:, 0, 1:, :] + root.unsqueeze(1)).logsumexp(-1).sum()

        nt_marginal = torch.autograd.grad(log_prefix_prob, indicator, retain_graph=True, create_graph=True)[0]

        return {'marginal': nt_marginal,
                'partition': (x.squeeze(1) + root).logsumexp(-1)
                }

        # # (b, N-1)
        # prefix_prob = (pi[:, 0, 1:, :] + root.unsqueeze(1)).logsumexp(-1)
        # prefix_prob2 = prefix_prob.new_zeros(*prefix_prob.shape)
        # prefix_prob2[:, 1:] = prefix_prob[:, :-1]
        # next_word_prob = prefix_prob - prefix_prob2
        # next_word_prob = next_word_prob.sum(-1)
        # stop_prob =  (logZ - prefix_prob[:,-1])
        # return {'partition': next_word_prob + stop_prob}
        
        
        
            

    

       


    @torch.enable_grad()
    def _inside2(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None):
        assert viterbi is not True
        # B, L, r_p
        unary = rules['unary']
    
        # B, L, r_m
        root = rules['root']

        # r_m, r_m 
        L = rules['left_m'].contiguous()
        R = rules['right_m'].contiguous()
        P = rules['P'].contiguous()

        split = rules['split']
        # r_p, r_p
        # L_p = L 
        # R_p = R 
        # LR = torch.cat([L, R], dim=-1)

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

        unary = unary + split[:, 0][None, None, :]
        split = split[:, 1].contiguous()

        unary_max = unary.max(-1)[0]
        
        unary = (unary - unary_max.unsqueeze(-1)).exp()        
        left_s = unary.new_zeros(batch, N, N, r_m)
        right_s = unary.new_zeros(batch, N, N, r_m)
        pi = unary.new_zeros(batch, N, N, r_m)
        pi_need_parent = unary.new_zeros(batch, N, N, r_m)

        diagonal_copy_(
            left_s, 
             (torch.einsum('bnp, pq -> bnq',  unary , L) + 1e-9).log() + unary_max.unsqueeze(-1),
             1
        )

        diagonal_copy_(
            right_s,
            (torch.einsum('bnp, pq -> bnq', unary, R) + 1e-9).log() + unary_max.unsqueeze(-1),
            1
        )

        unary = unary @ P 

        diagonal_copy_(
            pi, 
            (unary + 1e-9).log() + unary_max.unsqueeze(-1), 1
        )
        
        diagonal_copy_(
            pi_need_parent,
            ((unary @ R) + 1e-9).log() + unary_max.unsqueeze(-1), 1
        )

        for w in range(2, N):
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            W = stripe(pi_need_parent, n, (w-1), (1,w), 0)
            x = (Y + Z).logsumexp(-2) + span_indicator[:, torch.arange(n), w + torch.arange(n)].unsqueeze(-1) + split[None, None, :]
            
            if w + 1 < N:
                x_max = x.max(-1)[0]
                x = (x - x_max.unsqueeze(-1)).exp()                
                diagonal_copy_(left_s, (x @ L + 1e-9).log() + x_max.unsqueeze(-1) , w)
                diagonal_copy_(right_s, (x @ R + 1e-9).log() + x_max.unsqueeze(-1), w)
            
            x_pi = (Y + W).logsumexp(-2)
            x_pi_max = x_pi.max(-1)[0]
            x_pi = (x_pi - x_pi_max.unsqueeze(-1)).exp()
            x_pi = x_pi @ P
            
            diagonal_copy_(
                pi, (x_pi + 1e-9).log() + x_pi_max.unsqueeze(-1), w
            )

            if w + 1 < N:
                x_pi_need_parent = x_pi @ R             
                diagonal_copy_(
                    pi_need_parent, (x_pi_need_parent + 1e-9).log() + x_pi_max.unsqueeze(-1), w
                )

        logZ = (x.squeeze(1) + root).logsumexp(-1)
        # (b, N-1)
        prefix_prob = (pi[:, 0, 1:, :] + root.unsqueeze(1)).logsumexp(-1)
        prefix_prob2 = prefix_prob.new_zeros(*prefix_prob.shape)
        prefix_prob2[:, 1:] = prefix_prob[:, :-1]
        
        next_word_prob = prefix_prob - prefix_prob2
        next_word_prob = next_word_prob.sum(-1)
        stop_prob =  (logZ - prefix_prob[:,-1])        

        return {'partition': next_word_prob + stop_prob}
        

        
            
        

        
        
        
            

    

