import torch
from parser.pcfgs.fn import  stripe, diagonal_copy_, diagonal

class PCFG_base():

    def _inside(self):
        raise NotImplementedError

    def inside(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def decode(self, rules, lens, viterbi=False, mbr=False):
        return self._inside(rules=rules, lens=lens, viterbi=viterbi, mbr=mbr)


    def _get_prediction(self, logZ, span_indicator, lens, mbr=False):
        batch, seq_len = span_indicator.shape[:2]
        prediction = [[] for _ in range(batch)]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            assert logZ.requires_grad
            logZ.sum().backward()
            marginals = span_indicator.grad
            if mbr:
                return self._cky_zero_order(marginals.detach(), lens)
            else:
                viterbi_spans = marginals.nonzero().tolist()
                for span in viterbi_spans:
                    prediction[span[0]].append((span[1], span[2]))
        return prediction


    @torch.no_grad()
    def _cky_zero_order(self, marginals, lens):
        N = marginals.shape[-1]
        s = marginals.new_zeros(*marginals.shape).fill_(-1e9)
        p = marginals.new_zeros(*marginals.shape).long()
        diagonal_copy_(s, diagonal(marginals, 1), 1)
        for w in range(2, N):
            n = N - w
            starts = p.new_tensor(range(n))
            if w != 2:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            else:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            X, split = (Y + Z).max(2)
            x = X + diagonal(marginals, w)
            diagonal_copy_(s, x, w)
            diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)

        def backtrack(p, i, j):
            if j == i + 1:
                return [(i, j)]
            split = p[i][j]
            ltree = backtrack(p, i, split)
            rtree = backtrack(p, split, j)
            return [(i, j)] + ltree + rtree

        p = p.tolist()
        lens = lens.tolist()
        spans = [backtrack(p[i], 0, length) for i, length in enumerate(lens)]
        return spans

    def get_plus_semiring(self, viterbi):
        if viterbi:
            def plus(x, dim):
                return torch.max(x, dim)[0]
        else:
            def plus(x, dim):
                return torch.logsumexp(x, dim)
        return plus

    def _eisner(self, attach, lens):
        self.huge = -1e9
        self.device = attach.device
        '''
        :param attach: The marginal probabilities.
        :param lens: sentences lens
        :return: predicted_arcs
        '''
        A = 0
        B = 1
        I = 0
        C = 1
        L = 0
        R = 1
        b, N, *_ = attach.shape
        attach.requires_grad_(True)
        alpha = [
            [
                [torch.zeros(b, N, N, device=self.device).fill_(self.huge) for _ in range(2)] for _ in range(2)
            ] for _ in range(2)
        ]
        alpha[A][C][L][:, :, 0] = 0
        alpha[B][C][L][:, :, -1] =  0
        alpha[A][C][R][:, :, 0] =  0
        alpha[B][C][R][:, :, -1] = 0
        semiring_plus = self.get_plus_semiring(viterbi=True)
        # single root.
        start_idx = 1
        for k in range(1, N-start_idx):
            f = torch.arange(start_idx, N - k), torch.arange(k+start_idx, N)
            ACL = alpha[A][C][L][:, start_idx: N - k, :k]
            ACR = alpha[A][C][R][:,  start_idx: N - k, :k]
            BCL = alpha[B][C][L][:,  start_idx+k:, N - k:]
            BCR = alpha[B][C][R][:,  start_idx+k:, N - k :]
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_l = x + attach[:, f[1], f[0]]
            alpha[A][I][L][:,  start_idx: N - k, k] = arcs_l
            alpha[B][I][L][:, k+start_idx:N, N - k - 1] = arcs_l
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_r = x + attach[:, f[0], f[1]]
            alpha[A][I][R][:, start_idx: N - k, k] = arcs_r
            alpha[B][I][R][:, k+start_idx:N, N - k - 1] = arcs_r
            AIR = alpha[A][I][R][:, start_idx: N - k, 1 : k + 1]
            BIL = alpha[B][I][L][:, k+start_idx:, N - k - 1 : N - 1]
            new = semiring_plus(ACL + BIL, dim=2)
            alpha[A][C][L][:, start_idx: N - k, k] = new
            alpha[B][C][L][:, k+start_idx:N, N - k - 1] = new
            new = semiring_plus(AIR + BCR, dim=2)
            alpha[A][C][R][:, start_idx:N-k, k] = new
            alpha[B][C][R][:, start_idx+k:N, N - k - 1] = new
        # dealing with the root.
        root_incomplete_span = alpha[A][C][L][:, 1, :N-1] + attach[:, 0, 1:]
        for k in range(1,N):
            AIR = root_incomplete_span[:, :k]
            BCR = alpha[B][C][R][:, k, N-k:]
            alpha[A][C][R][:, 0, k] = semiring_plus(AIR+BCR, dim=1)
        logZ = torch.gather(alpha[A][C][R][:, 0, :], -1, lens.unsqueeze(-1))
        arc = torch.autograd.grad(logZ.sum(), attach)[0].nonzero().tolist()
        predicted_arc = [[] for _ in range(logZ.shape[0])]
        for a in arc:
            predicted_arc[a[0]].append((a[1] - 1, a[2] -1 ))
        return predicted_arc

