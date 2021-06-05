from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import  stripe, diagonal_copy_, checkpoint

import torch

class TDPCFG(PCFG_base):
    def __init__(self):
        super(TDPCFG, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False):
        assert viterbi is not True
        unary = rules['unary']
        root = rules['root']

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules['head']  # (batch, NT, r) r:=rank
        L = rules['left']  # (batch, NT+T, r)
        R = rules['right'] # (batch, NT+T, r)

        T = unary.shape[-1]
        S = L.shape[-2]
        NT = S - T
        # r = L.shape[-1]

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        @checkpoint
        def transform_left_t(x, left):
            '''
            :param x: shape (batch, n, T)
            :return: shape (batch, n, r)
            '''
            return (x.unsqueeze(-1) + left.unsqueeze(1)).logsumexp(2)

        @checkpoint
        def transform_left_nt(x, left):
            return (x.unsqueeze(-1) + left.unsqueeze(1)).logsumexp(2)

        @checkpoint
        def transform_right_t(x, right):
            return (x.unsqueeze(-1) + right.unsqueeze(1)).logsumexp(2)

        @checkpoint
        def transform_right_nt(x, right):
            return (x.unsqueeze(-1) + right.unsqueeze(1)).logsumexp(2)

        # @checkpoint
        def merge(Y, Z):
            '''
            :param Y: shape (batch, n, w, r)
            :param Z: shape (batch, n, w, r)
            :return: shape (batch, n, x)
            '''
            # contract dimension w.
            b_n_r = (Y + Z).logsumexp(-2)
            # contract dimension r.
            b_n_x = (b_n_r.unsqueeze(-2) + H.unsqueeze(1)).logsumexp(-1)
            return b_n_x


        batch, N, *_ = unary.shape
        N += 1

        # for estimating marginals.
        span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)

        left_term = transform_left_t(unary,L_term)
        right_term = transform_right_t(unary,R_term)

        s = unary.new_zeros(batch, N, N, NT).fill_(-1e9)
        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)

        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # w: span width
        for w in range(2, N):
            # n: the number of spans of width w.
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            x = merge(Y.clone(), Z.clone())
            x = x + span_indicator[:, torch.arange(n), w + torch.arange(n)].unsqueeze(-1)
            if w + 1 < N:
                left_x = transform_left_nt(x,L_nonterm)
                right_x = transform_right_nt(x, R_nonterm)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
            diagonal_copy_(s, x, w)

        final = s[torch.arange(batch), 0, lens] + root
        logZ = final.logsumexp(-1)

        if not mbr and not viterbi:
            return {'partition': logZ}

        else:

            return {
                    "prediction" : self._get_prediction(logZ, span_indicator, lens, mbr=True),
                    "partition" : logZ
                    }







