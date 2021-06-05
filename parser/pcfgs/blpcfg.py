import torch
from .pcfgs import PCFG_base
from .fn import checkpoint, stripe, diagonal_copy_, stripe_grammar_rules, stripe_with_headword, diagonal_copy_with_headword
from .eisner_satta import EisnerSatta

class BLPCFG(PCFG_base):

    @torch.enable_grad()
    def _inside(self, rules, lens, decoding=False):


        noninherent = rules['noninherent']
        inherent = rules['inherent']
        root = rules['root']
        head = rules['head']

        # B, D, H, NT, S, _ = rule.shape
        # B: batch_size D: direction, H: head word \alpha, NT: head-nt, S: NT+T
        H = root.shape[1]
        NT = root.shape[-1]
        S = noninherent.shape[2]
        r = inherent.shape[-1]
        B = root.shape[0]
        N = H + 1
        nt_slice = slice(0, NT)
        t_slice = slice(NT, S)

        s = head.new_zeros(B, N, N, H, NT).fill_(-1e9)


        s_noninherent = head.new_zeros(B, N, N, r, 2).fill_(-1e9)
        s_inherent = head.new_zeros(B, N, N, H, r).fill_(-1e9)

        noninherent_nt = noninherent[:, :, nt_slice, :].contiguous()
        inherent_nt = inherent[:, nt_slice].contiguous()

        LEFT = 0
        RIGHT = 1

        s_noninherent[:, torch.arange(H), torch.arange(H) + 1, :] = noninherent[:, :, t_slice, :, :].logsumexp(-3)

        s_inherent[:, torch.arange(H), torch.arange(H) + 1, torch.arange(H)] = inherent[:, t_slice, :].logsumexp(-2).unsqueeze(1).expand(-1, H, -1)

        span_indicator = head.new_zeros(B, N, N).requires_grad_(decoding)

        # calculation of s_{i,j}^{A,p} [Term A1 (left) + Term A2 (right)] in Equation (3) of the paper.
        @checkpoint
        def merge(left, right, closed_left, closed_right, head_rule):
            '''
            :param left: (b, n, w, h, r)
            :param right: (b, n, w, h, r)
            :param closed_left: (b, n, w, r)
            :param closed_right: (b, n, w, r)
            :param head_rule: (b, n, w, h, NT, r)
            :return: inside score (b, n, h, NT)
            '''
            # contract w firstly
            # 0=left, 1=right
            left = ((left + closed_right[..., None, :]).logsumexp(-3)[..., None, :] )
            right = ((right + closed_left[..., None, :]).logsumexp(-3)[..., None, :] )
            # combine left and right case.
            headed =  torch.stack([left, right]).logsumexp([0])
            # contract tensor-rank r.
            return  (headed + head_rule).logsumexp(-1)

        # Term D1-2 in the paper.
        def contract_qC(s, rule):
            '''
            contract the head word and nt of the noninherent span.
            :param s:  inside score of shape (b, n, h, NT)
            :param rules: of shape (b, n, h, NT, r, D)  [ log p( H-> beta) + log P(H->C,D) ] noninherent nt + beta
            :return: (b, n, r)
            '''
            return (s[..., None, None] + rule).logsumexp([-3, -4])

        # Term D-1-1 in the paper
        def contract_B(s, rule):
            '''
            :param s: inside score of shape (b, n, h, NT)
            :param rule: shape (b, NT, r),  log p(H -> B) inherent nt.
            :return: shape (b, n, h, r)
            '''
            return (s[..., None] + rule[:, None, None, ...]).logsumexp(-2)


        for w in range(2, N):
            n = N - w

            # Equation (8).  Term D1-1 \times Term D1-2
            # (b, n, w, H, r)
            left = stripe_with_headword(s_inherent, n, w - 1, (0, 1))
            right = stripe_with_headword(s_inherent, n, w - 1, (1, w), 0)
            # (b, n, w, r)
            closed_left = stripe(s_noninherent[..., LEFT], n, w - 1, (0, 1))
            closed_right = stripe(s_noninherent[..., RIGHT], n, w - 1, (1, w), 0)
            # (b, n, H, A)
            headed = merge(left.clone(),  right.clone(),closed_left.clone(), closed_right.clone(), stripe_grammar_rules(head, n, w))
            headed = headed + span_indicator[:, torch.arange(n), torch.arange(n) + w, None, None]
            diagonal_copy_with_headword(s, headed, w)

            if w < N - 1:
                # calculating Term D1-2 as described in the paper.
                headed_closed = contract_qC(headed, stripe_grammar_rules(noninherent_nt, n, w))
                # calculating Term D1-1 as described in the paper.
                headed = contract_B(headed, inherent_nt)
                # caching them.
                diagonal_copy_(s_noninherent, headed_closed, w)
                diagonal_copy_with_headword(s_inherent, headed, w)

        logZ = (s[torch.arange(B), 0, lens] + root).logsumexp([-1, -2])


        if not decoding:
            if lens.max() > 1:
                return {'partition': logZ}
            else:
                return {'partition': 0}

        else:
            return {'prediction': self._get_prediction(logZ, span_indicator, lens, mbr=True),
                    'partition': logZ
                    }

    def loss(self, rules, lens):
        return self._inside(rules, lens, decoding=False)

    def decode(self, rules, lens, mbr=True, viterbi=False, eval_dep=False):
        # It seems no way but to recover the origin tensor and perform the Eisner-Satta alg.
        if not eval_dep:
            return self._inside(rules, lens, decoding=True)
        else:
            prediction = self._inside(rules, lens, decoding=True)
            prediction['prediction_arc'] = self._mbr_arc(rules, lens)
            return prediction


    # I did not find an elegant way to estimate the marginals of arcs as that of spans. sorry for the messy code below.
    # Basically, codes below did not fully make use of the ```fold-unfold`` technique in order to make a `direct` connection between head word and child word to
    # estimate the marginals of arcs using automatic differentials. More efficient way may exist.
    # Warning: mbr decoding for dependencies may need MUCH more GPU memories and much slower as I did not optimize the codes.
    # So I suggest not evaluating this during training and only evaluate for the final trained models w. batch size 1.
    def _mbr_arc(self, rules, lens):
        noninherent = rules['noninherent']
        inherent = rules['inherent']
        root = rules['root']
        head = rules['head']
        # B, D, H, NT, S, _ = rule.shape
        H = root.shape[1]
        NT = root.shape[-1]
        S = noninherent.shape[2]
        r = inherent.shape[-1]
        B = root.shape[0]
        N = H + 1
        # print(NT, N, S)
        nt_slice = slice(0, NT)
        t_slice = slice(NT, S)

        s = torch.zeros(B, N, N, H, NT, device=torch.device("cuda")).fill_(-1e9)

        s_noninherent_left = torch.zeros(B, N, N, H, r, device=torch.device("cuda")).fill_(-1e9)
        s_noninherent_right = torch.zeros(B, N, N, H, r, device=torch.device("cuda")).fill_(-1e9)
        s_inherent = torch.zeros(B, N, N, H, r, device=torch.device('cuda')).fill_(-1e9)

        with torch.enable_grad():
            arc_indicator = torch.zeros(B, H, H, device=torch.device('cuda')).requires_grad_(True)
            root = root.detach().requires_grad_(True)

            LEFT = 0

            RIGHT = 1
            s_noninherent_left[:, torch.arange(H), torch.arange(H) + 1, :, :] = noninherent[:, :, t_slice, :, LEFT].logsumexp(
                -2).unsqueeze(-2) + arc_indicator[:,  torch.arange(H), :].unsqueeze(-1)
            s_noninherent_right[:, torch.arange(H), torch.arange(H) + 1, :,  :] = noninherent[:, :, t_slice, :,
                                                                              RIGHT].logsumexp(-2).unsqueeze(-2) + arc_indicator[:, torch.arange(H), :].unsqueeze(-1)
            s_inherent[:, torch.arange(H), torch.arange(H) + 1, torch.arange(H)] = inherent[:, t_slice, :].logsumexp(
                -2).unsqueeze(1).expand(-1, H, -1)

            @checkpoint
            def reduce_1(left, closed_left, right, closed_right):
                left = (left[:, :, :, :] + closed_right[:, :, :, :, :]).logsumexp([2])
                right = (right[:, :, :,  :] + closed_left[:, :, :, :, :]).logsumexp([2])
                return torch.stack([left, right]).logsumexp([0])

            @checkpoint
            def reduce_2(a, b):
                return (a + b).logsumexp(-1)

            @checkpoint
            def reduce_3(a, b):
                return (a + b).logsumexp(-2)


            for w in range(2, N):
                n = N - w

                if w == 2:
                    headed = torch.zeros(B, n, w, NT, device=torch.device("cuda")).fill_(-1e9)

                    headed[:, :, LEFT, ...] = (
                            s_noninherent_right[:, torch.arange(n) + 1, torch.arange(n) + 2, torch.arange(n),  None, :] +\
                            s_inherent[:, torch.arange(n), torch.arange(n) + 1, torch.arange(n),  None, :] +\
                            head[:, torch.arange(n), :, :]
                    ).logsumexp(-1)

                    headed[:, :, RIGHT, ...] = (
                            s_noninherent_left[:, torch.arange(n), torch.arange(n) + 1, torch.arange(n) + 1, None, :] +\
                            s_inherent[:, torch.arange(n) + 1, torch.arange(n) + 2, torch.arange( n) + 1, None, :] +\
                            head[:,  torch.arange(n) + 1,  :,  :]
                        ).logsumexp(-1)

                else:
                    left = stripe_with_headword(s_inherent, n, w - 1, (0, 1))
                    right = stripe_with_headword(s_inherent, n, w - 1, (1, w), 0)
                    closed_left = stripe_with_headword(s_noninherent_left, n, w - 1, (0, 1))
                    closed_right = stripe_with_headword(s_noninherent_right, n, w - 1, (1, w), 0)
                    # b, n, w, w+1,
                    headed_tmp = reduce_1(left.clone(), closed_left.clone(), right.clone(),
                                            closed_right.clone())
                    headed = reduce_2(headed_tmp[..., None, :], stripe_grammar_rules(head, n, w))

                # headed = headed + span_indicator[:, torch.arange(n), torch.arange(n) + w, None, None]
                diagonal_copy_with_headword(s, headed, w)

                if w < N - 1:

                    # headed: (batch, n, child, nt)
                    # noninherent; (batch, child, nt, r, direction),  after stripe->  (batch, n, child, nt, r, direction)
                    # arcindicator: (batch, n, child, HEAD)
                    #  (batch, n, child, 1, nt, 1, 1) + (batch, n, child, 1, nt, r, direction) -> (batch, n, child, 1, r, direction) [marginalzie over nt]
                    #  + (batch, n, child, HEAD, 1, 1)
                    #  -> (batch, n, child, HEAD,, r, direction) -> (batch, n, HEAD, r, direction) [marginalize over the non-inherenting span's head index]
                    @checkpoint
                    def reduce22(a, b, c):
                        # marginalize nt first
                        tmp = (a[:, :, :, None, :, None, None ] + b[:, :, :, None, :, :, :]).logsumexp(-3)
                        return (tmp + c[..., None, None]).logsumexp(2)

                    headed_closed = reduce22(headed,
                                               stripe_grammar_rules(noninherent[:, :, nt_slice, :, ], n, w),
                                               stripe_grammar_rules(arc_indicator, n, w)
                                               )
                    headed = reduce_3(headed[..., None], inherent[:, None, None, nt_slice])
                    diagonal_copy_(s_noninherent_left, headed_closed[..., 0], w)
                    diagonal_copy_(s_noninherent_right, headed_closed[..., 1], w)
                    diagonal_copy_with_headword(s_inherent, headed, w)

            logZ = (s[:, 0, -1]) + root
            logZ = logZ.logsumexp([-1, -2])
            logZ.sum().backward()
            arc_marginals = arc_indicator.grad
            root_attach = root.grad
            root_attach = root_attach.sum(2)
            attach = torch.zeros(B, N, N, device=torch.device("cuda")).fill_(-1e9)
            attach[:, 0, 1:] = root_attach
            attach[:, 1:, 1:] = arc_marginals.transpose(-1, -2)
            del s, s_noninherent_left, s_inherent, s_noninherent_right
            arc_prediction = self._eisner(attach, lens)
            return  arc_prediction
