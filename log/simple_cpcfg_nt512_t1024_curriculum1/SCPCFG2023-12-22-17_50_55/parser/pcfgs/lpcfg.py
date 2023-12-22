import torch
from parser.pcfgs.pcfgs import PCFG_base
from torch.utils.checkpoint import checkpoint as ckp

'''
Refactor this file would kill me. It can run anyway.
This is a reimplementation of the inside algorithm of (Zhu et al 2020)
'''

class L_PCFG(PCFG_base):
    def __init__(self):
        super(L_PCFG, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    def decode(self, rules, lens, mbr=True, viterbi=False, eval_dep=False):
        if not eval_dep:
            return self._inside_v2(rules, lens, decoding=True)
        else:
            return self._mbr(rules, lens)

    # quadractic in m.
    @torch.enable_grad()
    def _inside(self, rules, lens, **kwargs):
        unary = rules['unary']
        root = rules['root']
        rule = rules['rule']
        logZ = lexicalizedPCFG.apply(unary, rule, root)
        return {'partition' : logZ}
    
    # parallel impl. but cubic in m

    @torch.enable_grad()
    def _inside_v2(self, rules, lens, decoding=False):
        unary = rules['unary']
        root = rules['root']
        rule = rules['rule']
        # , N, _ = unary.shape
        B, D, N, NT, S, _ = rule.shape
        T = S - NT
        nt_slice = slice(0, NT)
        t_slice = slice(NT, S)
        unary_t = unary[..., t_slice].contiguous()
        unary_nt = unary[..., nt_slice].contiguous()
        N += 1
        LEFT = 0
        RIGHT = 1
        beta = torch.zeros(B, N, N, N - 1, NT, device=torch.device("cuda")).fill_(-1e9)
        beta_closed = torch.zeros(B, N, N, NT, device=torch.device('cuda')).fill_(-1e9)
        left_rule = rule[:, LEFT, :, :, nt_slice, nt_slice].contiguous()
        right_rule = rule[:, RIGHT, :, :, nt_slice, nt_slice].contiguous()
        # _term: headed by t.  _nonterm: headed by nt.
        right_rule_term = rule[:, RIGHT, :, :, nt_slice, t_slice].contiguous()
        right_rule_nonterm = rule[:, RIGHT, :, :, t_slice, nt_slice].contiguous()
        left_rule_term = rule[:, LEFT, :, :, t_slice, nt_slice].contiguous()
        left_rule_nonterm = rule[:, LEFT, :, :, nt_slice, t_slice].contiguous()

        span_indicator = root.new_zeros(B, N, N).requires_grad_(decoding)


        def reduce_lasttwo_ab(a, b):
            return (a + b).logsumexp([-1, -2])

        def reduce_lasttwo_abc(a, b, c):
            return (a + b + c).logsumexp([-1, -2])

        for w in range(2, N):
            n = N - w
            Y_term = unary_t[:, :n, ...]
            Z_term = unary_t[:, w - 1:, ...]
            if w == 2:
                headed = torch.zeros(B, n, 2, NT, T, T, device=torch.device('cuda')).fill_(-1e9)
                headed[:, :, LEFT, ...] = Z_term[:, :, None, None, :] + rule[:, LEFT, :n, :, t_slice, t_slice]
                headed[:, :, RIGHT, ...] = Y_term[:, :, None, :, None] + rule[:, RIGHT, w - 1:, :, t_slice, t_slice]
                headed = headed.logsumexp([-1, -2]) + span_indicator[:, torch.arange(n), w+torch.arange(n), None, None]
                headed_closed = (headed + stripe_grammar_rules(unary_nt, n, w)).logsumexp(2)
                diagonal_copy_v2(beta, headed, w)
                diagonal_copy_(beta_closed, headed_closed, w)
                continue

            Y_closed = stripe(beta_closed, n, w - 1, (0, 1)).clone()
            Z_closed = stripe(beta_closed, n, w - 1, (1, w), 0).clone()

            if w > 3:
                headed = torch.zeros(2, B, n, w, NT, device=torch.device('cuda')).fill_(-1e9)
                y = stripe_version_nt_nt(beta, n, w - 1, (0, 1))[..., None, :, None]
                z = Z_closed[:, :, 1:-1, None, None, None, :]
                yz = (y + z).logsumexp(2)
                # print("fuck", yz)
                headed[0, :, :, :-2] = ckp(reduce_lasttwo_ab, yz, stripe_grammar_rules(left_rule, n, w - 2))
                y = Y_closed[:, :, 1:-1, None, None, :, None]
                z = stripe_version_nt_nt(beta, n, w - 1, (1, w), 0)[..., None, None, :]
                yz = (y + z).logsumexp(2)
                # print("fuck", yz)
                headed[1, :, :, 2:] = ckp(reduce_lasttwo_ab, yz,
                                                 stripe_grammar_rules(right_rule, n, w - 2, offset=2))
                x_0 = headed.logsumexp(0)

            y = Y_closed[:, :, -1, ...]
            z = Z_term

            headed_left = ckp(reduce_lasttwo_abc,
                                     stripe_version_nt_t(beta, n, w - 1)[..., None, :, None].clone(),
                                     stripe_grammar_rules(left_rule_nonterm, n, w - 1),
                                     z[..., None, None, None, :])

            headed_right = ckp(reduce_lasttwo_ab,
                                      y[..., None, None, :, None],
                                      stripe_grammar_rules(right_rule_term, n, 1, offset=w - 1))
            x_1 = torch.cat([headed_left, headed_right], dim=2)
            y = Y_term
            z = Z_closed[:, :, 0, ...]
            headed_left = (z[..., None, None, None, :] + stripe_grammar_rules(left_rule_term, n, 1)).logsumexp([-1, -2])
            headed_right = ckp(reduce_lasttwo_abc,
                                      stripe_version_t_nt(beta, n, w - 1)[..., None, None, :].clone(),
                                      stripe_grammar_rules(right_rule_nonterm, n, w - 1, offset=1),
                                      y[..., None, None, :, None])
            x_2 = torch.cat([headed_left, headed_right], dim=2)
            if w == 3:
                x = torch.stack([x_1, x_2])
            else:
                x = torch.stack([x_0, x_1, x_2])
            x = x.logsumexp(0)
            x = x + span_indicator[:, torch.arange(n), w+torch.arange(n), None, None]
            x_closed = (x + stripe_grammar_rules(unary_nt, n, w)).logsumexp(2)
            # if w < N-1:
            diagonal_copy_v2(beta, x, w)
            diagonal_copy_(beta_closed, x_closed, w)
        final = beta_closed[:, 0, -1, ...] + root
        logZ = final.logsumexp(-1)

        if not decoding:
            return {'partition': logZ}
        else:
            return {'prediction': self._get_prediction(logZ, span_indicator, lens, mbr=True),
                    'partition': logZ
                    }

    def _mbr(self, rules, lens, **kwargs):
        if lens.max() == 1:
            return {'prediction' : [[] for _ in range(lens.shape[0])],
                    'prediction_arc': [[(-1, 0)] for _ in range(lens.shape[0])]}

        unary = rules['unary']
        root = rules['root']
        rule = rules['rule']

        # , N, _ = unary.shape
        B, D, N, NT, S, _ = rule.shape
        nt_slice = slice(0, NT)
        t_slice = slice(NT, S)
        unary_t = unary[..., t_slice].contiguous()
        unary_nt = unary[..., nt_slice].contiguous()
        N += 1
        H = N - 1
        T = S - NT
        LEFT = 0
        RIGHT = 1
        beta = torch.zeros(B, N, N, N - 1, NT, device=torch.device("cuda")).fill_(-1e9)
        beta_closed = torch.zeros(B, N, N, H, NT, device=torch.device('cuda')).fill_(-1e9)
        left_rule = rule[:, LEFT, :, :, nt_slice, nt_slice].contiguous()
        right_rule = rule[:, RIGHT, :, :, nt_slice, nt_slice].contiguous()
        # _term: headed by t.  _nonterm: headed by nt.
        right_rule_term = rule[:, RIGHT, :, :, nt_slice, t_slice].contiguous()
        right_rule_nonterm = rule[:, RIGHT, :, :, t_slice, nt_slice].contiguous()
        left_rule_term = rule[:, LEFT, :, :, t_slice, nt_slice].contiguous()
        left_rule_nonterm = rule[:, LEFT, :, :, nt_slice, t_slice].contiguous()

        with torch.enable_grad():
            arc_indicator = torch.zeros(B, N-1, N-1, device=torch.device("cuda")).requires_grad_(True)
            span_indicator = torch.zeros(B, N, N, device=torch.device('cuda')).requires_grad_(True)
            root = root.unsqueeze(1).expand(B, H, NT).detach().clone().requires_grad_(True)

            final = None

            def reduce_lasttwo_a(a):
                return a.logsumexp([-1, -2])

            def reduce_lasttwo_ab(a, b):
                return (a + b).logsumexp([-1, -2])

            def reduce_lasttwo_abc(a, b, c):
                return (a + b + c).logsumexp([-1, -2])

            for w in range(2, N):
                n = N - w
                Y_term = unary_t[:, :n, ...]
                Z_term = unary_t[:, w - 1:, ...]
                if w == 2:
                    headed = torch.zeros(B, n, 2, NT, T, T, device=torch.device('cuda')).fill_(-1e9)
                    # print( Z_term[:, :, None, None, :].shape, rule[:, LEFT, :n, :, t_slice, t_slice].shape, arc_indicator[:, torch.arange(N) + 1, torch.arange(N).shape )

                    headed[:, :, LEFT, ...] = Z_term[:, :, None, None, :] + rule[:, LEFT, :n, :, t_slice, t_slice] + arc_indicator[:, torch.arange(n) + 1, torch.arange(n), None, None, None]
                    headed[:, :, RIGHT, ...] = Y_term[:, :, None, :, None] + rule[:, RIGHT, w - 1:, :, t_slice, t_slice] + arc_indicator[:, torch.arange(n), torch.arange(n) + 1 , None, None , None]
                    headed = headed.logsumexp([-1, -2])
                    headed = headed + span_indicator[:, torch.arange(n), torch.arange(n) + w, None, None]

                    # headed: size? ( batch, n, w,  NT)
                    if lens.max() > 2:
                        headed_closed = (headed.unsqueeze(-2) +
                                         stripe_grammar_rules(unary_nt, n, w).unsqueeze(-2) +
                                         stripe_grammar_rules(arc_indicator, n, w).unsqueeze(-1)
                                         ).logsumexp(2)

                        diagonal_copy_v2(beta, headed, w)
                        diagonal_copy_(beta_closed, headed_closed, w)
                    else:
                        final = (headed + stripe_grammar_rules(unary_nt, n, w))


                    continue

                Y_closed = stripe_version2(beta_closed, n, w - 1, (0, 1)).clone()
                Z_closed = stripe_version2(beta_closed, n, w - 1, (1, w), 0).clone()

                if w > 3:
                    headed = torch.zeros(2, B, n, w, NT, device=torch.device('cuda')).fill_(-1e9)
                    y = stripe_version_nt_nt(beta, n, w - 1, (0, 1))[..., None, :, None]
                    # print(y.shape)
                    z = Z_closed[:, :,  1:-1,  1:-1, None, None, :]
                    # print(z.shape)
                    # print(y.shape, z.shape, "??????")

                    yz = (y + z).logsumexp(2)
                    headed[0, :, :,  :-2] = ckp(reduce_lasttwo_ab, yz, stripe_grammar_rules(left_rule, n, w - 2))
                    y = Y_closed[:, :,   1:-1,  1:-1,  None, :, None]
                    z = stripe_version_nt_nt(beta, n, w - 1, (1, w), 0)[..., None, None, :]
                    yz = (y + z).logsumexp(2)

                    headed[1, :, :, 2:] = ckp(reduce_lasttwo_ab, yz,
                                                     stripe_grammar_rules(right_rule, n, w - 2, offset=2))
                    x_0 = headed.logsumexp(0)

                y = Y_closed[:, :, -1, None, -1, ...]
                z = Z_term

                headed_left = ckp(reduce_lasttwo_abc,
                                         stripe_version_nt_t(beta, n, w - 1)[..., None, :, None].clone(),
                                         stripe_grammar_rules(left_rule_nonterm, n, w - 1),
                                         z[..., None,  None, None, :])

                # print(stripe_grammar_rules(right_rule_term, n, 1, offset=w - 1).shape)
                # print(y.shape)
                headed_right = ckp(reduce_lasttwo_ab,
                                          y[...,  None, :, None],
                                          stripe_grammar_rules(right_rule_term, n, 1, offset=w - 1))
                # print(headed_right.shape, "suppose to 1.")

                x_1 = torch.cat([headed_left, headed_right], dim=2)

                y = Y_term

                z = Z_closed[:, :, 0, None, 0, ...]


                headed_left = (z[...,  None, None, :] + stripe_grammar_rules(left_rule_term, n, 1)).logsumexp(
                    [-1, -2])

                headed_right = ckp(reduce_lasttwo_abc,
                                          stripe_version_t_nt(beta, n, w - 1)[..., None, None, :].clone(),
                                          stripe_grammar_rules(right_rule_nonterm, n, w - 1, offset=1),
                                          y[...,  None, None, :, None])

                x_2 = torch.cat([headed_left, headed_right], dim=2)
                if w == 3:
                    x = torch.stack([x_1, x_2])
                else:
                    # print(x_0.shape,)
                    # print(x_1.shape)
                    # print(x_2.shape)
                    x = torch.stack([x_0, x_1, x_2])
                x = x.logsumexp(0)
                # print(x.shape, '?????')

                x = x + span_indicator[:, torch.arange(n), torch.arange(n) + w, None, None]

                if w < N - 1:
                    (x + stripe_grammar_rules(unary_nt, n, w)).logsumexp(2)
                    x_closed = (x.unsqueeze(-2) +
                                 stripe_grammar_rules(unary_nt, n, w).unsqueeze(-2) +
                                 stripe_grammar_rules(arc_indicator, n, w).unsqueeze(-1)
                                 ).logsumexp(2)
                    diagonal_copy_v2(beta, x, w)
                    diagonal_copy_(beta_closed, x_closed, w)
                else:
                    final = (x + stripe_grammar_rules(unary_nt, n, w))

            final = final.squeeze(1)
            final = final + root
            # print(final.shape)
            logZ = final.logsumexp([-1, -2])

            # print(logZ)
            logZ.sum().backward()
            marginals = span_indicator.grad
            arc_marginals = arc_indicator.grad
            root_attach = root.grad
            root_attach = root_attach.sum(-1)
            attach = torch.zeros(B, N, N, device=torch.device("cuda")).fill_(-1e9)
            attach[:, 0, 1:] = root_attach
            attach[:, 1:, 1:] = arc_marginals.transpose(-1, -2)
            del beta, beta_closed
            prediction = self._cky_zero_order(marginals.detach(), lens)
            arc_prediction = self._eisner(attach, lens)
            return {'partition': logZ,
                    'prediction': prediction,
                    'prediction_arc': arc_prediction}



class lexicalizedPCFG(torch.autograd.Function):

    @staticmethod
    def forward(ctx, unary, rule, root):
        root = root.squeeze(1)
        LEFT = 0
        RIGHT = 1
        B, D, H, NT, S, _ = rule.shape
        T = S - NT
        N = H + 1
        nt_slice = slice(0, NT)
        t_slice = slice(NT, S)
        s = torch.zeros(B, N, N, H, NT, device=torch.device("cuda")).fill_(-1e9)
        s_closed = torch.zeros(B, N, N, NT, device=torch.device("cuda")).fill_(-1e9)
        s_need_dad = torch.zeros(B, N, N, H, NT, NT, device=torch.device('cuda')).fill_(-1e9)
        rule_t = rule.transpose(-1, -2).contiguous()

        for i in range(N - 1):
            if i > 0:
                s_need_dad[:, i, i + 1, :i] = (
                            unary[:, i, None, None, None, t_slice] + rule[:, LEFT, :i, :, nt_slice, t_slice]).logsumexp(
                    -1)
            if i < N - 1:
                s_need_dad[:, i, i + 1, i + 1:] = (
                            unary[:, i, None, None, t_slice, None] + rule[:, RIGHT, i + 1:, :, t_slice,
                                                                     nt_slice]).logsumexp(-2)

        LEFT = 0
        RIGHT = 1

        for w in range(2, N):
            n = N - w
            if w == 2:
                headed = torch.zeros(B, n, w, NT, T, T, device=torch.device("cuda")).fill_(-1e9)
                headed[:, :, LEFT, ...] = (
                            unary[:, torch.arange(n) + 1, None, None, t_slice] + rule[:, LEFT, torch.arange(n), :,
                                                                                 t_slice, t_slice])
                headed[:, :, RIGHT, ...] = (
                            unary[:, torch.arange(n), None, t_slice, None] + rule[:, RIGHT, torch.arange(n) + 1, :,
                                                                             t_slice, t_slice])
                headed = headed.logsumexp([-1, -2])
                diagonal_copy_v2(s, headed, w)
            else:
                left = stripe_version2(s, n, w - 1, (0, 1))
                right = stripe_version2(s, n, w - 1, (1, w), 0)
                left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
                right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)
                left = (left[:, :, :, :, None, :] + right_need_dad).logsumexp([2, -1])
                right = (right[:, :, :, :, None, :] + left_need_dad).logsumexp([2, -1])
                headed = torch.stack([left, right]).logsumexp(0)
                # headed by left most element of the span. [i, i+1] + [i+1, j] -> [i, j], headed by element i.
                left = (s_closed[:, torch.arange(n) + 1, torch.arange(n) + w, None, None, :] + rule[:, LEFT,
                                                                                               torch.arange(n), :,
                                                                                               t_slice,
                                                                                               nt_slice]).logsumexp(
                    [-2, -1])
                # headed by right most element of the span.
                right = (s_closed[:, torch.arange(n), torch.arange(n) + w - 1, None, :, None] + rule[:, RIGHT,
                                                                                                torch.arange(n) + w - 1,
                                                                                                :, nt_slice,
                                                                                                t_slice]).logsumexp(
                    [-1, -2])
                headed[:, :, 0] = torch.logaddexp(headed[:, :, 0], left)
                headed[:, :, -1] = torch.logaddexp(headed[:, :, -1], right)
                diagonal_copy_v2(s, headed, w)
            headed = (headed + stripe_grammar_rules(unary[..., nt_slice], n, w)).logsumexp(2)
            diagonal_copy_(s_closed, headed, w)
            sss = torch.zeros(B, N - w, N - w - 1, NT, NT, NT, device=torch.device("cuda")).fill_(-1e9)
            if w < N - 1:
                for l in range(N - w):
                    r = w + l
                    if l > 0:
                        sss[:, l, :l] = (headed[:, l, None, None, None, :] + rule[:, LEFT, :l, :, nt_slice, nt_slice])
                    if r < N - 1:
                        sss[:, l, l:] = (
                                    headed[:, l, None, None, None, :] + rule_t[:, RIGHT, r:, :, nt_slice, nt_slice])
                sss = sss.logsumexp(-1)
                for l in range(N - w):
                    r = w + l
                    s_need_dad[:, l, r, :l] = sss[:, l, :l]
                    s_need_dad[:, l, r, r:] = sss[:, l, l:]

        s_final = s_closed[:, 0, -1] + root

        logZ = (s_final).logsumexp(-1)

        gradient_unary = torch.zeros(B, H, S, device=torch.device('cuda'))
        gradient_rule = torch.zeros(B, 2, H, NT, S, S, device=torch.device('cuda'))
        out = torch.zeros(B, N, N, H, NT, device=torch.device("cuda"))
        out_closed = torch.zeros(B, N, N, NT, device=torch.device("cuda"))
        out_need_dad = torch.zeros(B, N, N, H, NT, NT, device=torch.device('cuda'))
        gradient_root = (s_final - logZ.unsqueeze(-1)).exp_()
        out_closed[:, 0, -1, ] = (gradient_root.unsqueeze(-2) * (
                    s[:, 0, -1] + unary[:, :, nt_slice] - s_closed[:, 0, -1].unsqueeze(-2)).exp_()).sum(1)

        for w in range(N - 1, 0, -1):
            n = N - w
            headed_closed = diagonal(s_closed, w)
            headed = diagonal_v2(s, w)

            if w > 1:
                out_s_need_dad = torch.zeros(B, N - w, N - w - 1, NT, NT, NT, device=torch.device("cuda"))
                out_need_dad_gradient = torch.zeros(B, n, n - 1, NT, NT, device=torch.device('cuda'))
                for l in range(N - w):
                    r = l + w
                    if l > 0:
                        out_s_need_dad[:, l, :l, ...] = (rule[:, LEFT, :l, :, nt_slice, nt_slice] - s_need_dad[:, l, r,
                                                                                                    :l].unsqueeze(-1))
                        out_need_dad_gradient[:, l, :l] = out_need_dad[:, l, r, :l]
                    if r < N - 1:
                        out_s_need_dad[:, l, l:, ...] = (
                                    rule_t[:, RIGHT, r:, :, nt_slice, nt_slice] - s_need_dad[:, l, r, r:].unsqueeze(-1))
                        out_need_dad_gradient[:, l, l:] = out_need_dad[:, l, r, r:]

                out_s_need_dad.add_(headed_closed[:, :, None, None, None, :]).exp_().mul_(
                    out_need_dad_gradient.unsqueeze(-1))

                out_closed[:, torch.arange(N - w), torch.arange(N - w) + w] += out_s_need_dad.sum([-2, -3, -4])

                for l in range(N - w):
                    r = l + w
                    if l > 0:
                        gradient_rule[:, LEFT, :l, :, nt_slice, nt_slice] += out_s_need_dad[:, l, :l, ...]
                    if r < N - 1:
                        gradient_rule[:, RIGHT, r:, :, nt_slice, nt_slice] += out_s_need_dad[:, l, l:, ...].transpose(
                            -1, -2)

                out_gradient = (headed + stripe_grammar_rules(unary[..., nt_slice], n, w) - headed_closed.unsqueeze(
                    2)).exp_()

                out_gradient.mul_(diagonal(out_closed, w).unsqueeze(2))

                for l in range(N - w):
                    r = l + w
                    out[:, l, r, l:r] += out_gradient[:, l, ]
                    gradient_unary[:, l:r, nt_slice] += out_gradient[:, l, ]

            elif w == 1:

                for l in range(N - w):
                    if l > 1:
                        tmp = (unary[:, l, None, None, None, t_slice] + rule[:, LEFT, :l, :, nt_slice,
                                                                        t_slice] - s_need_dad[:, l, l + 1,
                                                                                   :l].unsqueeze(-1)).exp_().mul_(
                            out_need_dad[:, l, l + 1, :l].unsqueeze(-1))
                        gradient_unary[:, l, t_slice] += tmp.sum([-2, -3, -4])
                        gradient_rule[:, LEFT, :l, :, nt_slice, t_slice] += tmp

                    if (l + 1) < N - 2:
                        tmp = (unary[:, l, None, None, t_slice, None] + rule[:, RIGHT, l + 1:, :, t_slice,
                                                                        nt_slice] - s_need_dad[:, l, l + 1,
                                                                                    l + 1:].unsqueeze(-2)).exp_().mul_(
                            out_need_dad[:, l, l + 1, l + 1:].unsqueeze(-2))
                        gradient_unary[:, l, t_slice] += tmp.sum([-1, -3, -4])
                        gradient_rule[:, RIGHT, l + 1:, :, t_slice, nt_slice] += tmp

            b_n_h_x = diagonal_v2(out, w)

            if w > 2:
                left = stripe_version2(s, n, w - 1, (0, 1))
                right = stripe_version2(s, n, w - 1, (1, w), 0)
                left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
                right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)
                left = (left[:, :, :, :, None, :] + right_need_dad - headed.unsqueeze(2).unsqueeze(-1)).exp_().mul_(
                    b_n_h_x.unsqueeze(2).unsqueeze(-1))
                right = (right[:, :, :, :, None, :] + left_need_dad - headed.unsqueeze(2).unsqueeze(-1)).exp_().mul_(
                    b_n_h_x.unsqueeze(2).unsqueeze(-1))

                stripe_version2_add(out, left.sum(-2), n, w - 1, (0, 1))
                stripe_version2_add(out_need_dad, left, n, w - 1, (1, w), 0)

                stripe_version2_add(out, right.sum(-2), n, w - 1, (1, w), 0)
                stripe_version2_add(out_need_dad, right, n, w - 1, (0, 1))

                left = (s_closed[:, torch.arange(n) + 1, torch.arange(n) + w, None, None, :] + rule[:, LEFT,
                                                                                               torch.arange(n), :,
                                                                                               t_slice,
                                                                                               nt_slice] - headed[:, :,
                                                                                                           0].unsqueeze(
                    -1).unsqueeze(-1)).exp_().mul_(b_n_h_x[:, :, 0].unsqueeze(-1).unsqueeze(-1))
                right = (s_closed[:, torch.arange(n), torch.arange(n) + w - 1, None, :, None] + rule[:, RIGHT,
                                                                                                torch.arange(n) + w - 1,
                                                                                                :, nt_slice,
                                                                                                t_slice] - headed[:, :,
                                                                                                           -1].unsqueeze(
                    -1).unsqueeze(-1)).exp_().mul_(b_n_h_x[:, :, -1].unsqueeze(-1).unsqueeze(-1))

                gradient_rule[:, LEFT, torch.arange(n), :, t_slice, nt_slice] += left
                gradient_rule[:, RIGHT, torch.arange(n) + w - 1, :, nt_slice, t_slice] += right

                out_closed[:, torch.arange(n) + 1, torch.arange(n) + w] += left.sum([-2, -3])
                out_closed[:, torch.arange(n), torch.arange(n) + w - 1] += right.sum([-1, -3])

            if w == 2:
                tmp = (unary[:, torch.arange(n), None, t_slice, None] + rule[:, RIGHT, torch.arange(n) + 1, :,
                                                                        t_slice, t_slice] - headed[:, :, 1,
                                                                                            ...].unsqueeze(
                    -1).unsqueeze(-1)).exp_().mul_(b_n_h_x[:, :, 1].unsqueeze(-1).unsqueeze(-1))

                gradient_rule[:, RIGHT, torch.arange(n) + 1, :, t_slice, t_slice] += tmp

                gradient_unary[:, torch.arange(n), t_slice] += tmp.sum([-1, -3])

                tmp = (unary[:, torch.arange(n) + 1, None, None, t_slice] + rule[:, LEFT, torch.arange(n), :, t_slice,
                                                                            t_slice] - headed[:, :, 0, ...].unsqueeze(
                    -1).unsqueeze(-1)).exp_().mul_(b_n_h_x[:, :, 0].unsqueeze(-1).unsqueeze(-1))
                gradient_rule[:, LEFT, torch.arange(n), :, t_slice, t_slice] += tmp
                gradient_unary[:, torch.arange(n) + 1, t_slice] += tmp.sum([-2, -3])

        ctx.rule_gradient = gradient_rule
        ctx.root_gradient = gradient_root
        ctx.unary_gradient = gradient_unary
        return logZ

    @staticmethod
    def backward(ctx, grad_output):
        multiplier = grad_output.max()
        # unary, rule, root
        return ctx.unary_gradient * multiplier, ctx.rule_gradient * multiplier, (ctx.root_gradient * multiplier)















# input: batch * child * head * nt * nt * nt
# output: batch * (n-1) * (w-1) * w * nt * nt * nt
def stripe_need_left_parent(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append((seq_len + 1) * numel)
    new_stride.extend(*stride[2:])
    return x.as_strided(size=(x.shape[0], n, *list(x.shape[3:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def stripe_left(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    # stride[1] = (seq_len + 1) * numel
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append((seq_len + 1) * numel)
    new_stride.extend(*stride[2:])
    return x.as_strided(size=(x.shape[0], n, *list(x.shape[3:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def stripe_right(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    # stride[1] = (seq_len + 1) * numel
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append((seq_len + 1) * numel)
    new_stride.extend(*stride[2:])

    return x.as_strided(size=(x.shape[0], n, *list(x.shape[3:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel) +  (w-1) * numel * (
               1 if dim == 0 else seq_len)

def stripe_proper(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w-2, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel + numel * (1 if dim == 0 else seq_len))
    else:
        raise NotImplementedError


    # else:
    #     return x.as_strided(size=(x.shape[0], n, w),
    #                         stride=stride,
    #                         storage_offset=(offset[0] * seq_len + offset[1]) * numel)





def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    else:
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)

def stripe_logadd1(x, value,  n, w,  offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
        x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                     stride=stride,
                     storage_offset=(offset[0] * seq_len + offset[1] + 1) * numel).copy_(torch.logaddexp(tmp, value))
    else:
        raise NotImplemented
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)




def logbmmexp(x, y):
    x = x.contiguous()
    y = y.contiguous()
    return (x.unsqueeze(-1) + y.unsqueeze(-3)).logsumexp(-2)

def maxbmm(x, y):
    return (x.unsqueeze(-1) + y.unsqueeze(-3)).max(-2)[0]





def stripe_grammar_rules(x, n, w, offset=0, addition = 0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, *x.shape[2:]),
                            stride=new_stride, storage_offset= offset*(stride[1]))






def stripe_version3(x, n, w, offset=0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=0)

def stripe_version5(x, n, w=0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[2:])),
                            stride=new_stride,
                            storage_offset=0)


def stripe_parent_left(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n,  w,  x.shape[3], t, nt),
                            stride=new_stride,
                            storage_offset=stride[2] + stride[-2]*nt)

def stripe_parent_left_add(x, y, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], n, w, x.shape[3], t, nt),
                 stride=new_stride,
                 storage_offset=stride[2] + stride[-2] * nt).add_(y)

def stripe_headed_left(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, x.shape[3], nt,  t),
                            stride=new_stride,
                            storage_offset= w * stride[2] + nt * stride[-1])


def stripe_rules_left(x, w, start, nt):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], w, start, nt, nt, nt),
                            stride=new_stride,
                            storage_offset= start * stride[1])


def stripe_rules_right(x, w, end, nt):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], w, seq_len - end , nt, nt, nt),
                            stride=new_stride,
                            storage_offset= end * stride[2])





def stripe_headed_left_add(x, y, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], n, w, x.shape[3], nt, t),
                 stride=new_stride,
                 storage_offset=w * stride[2] + nt * stride[-1]).add_(y)
    # return x


def stripe_headed_right(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, x.shape[3], t, nt),
                            stride=new_stride,
                            storage_offset= stride[1] + nt*stride[-2])

def stripe_headed_right_add(x, y, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], n, w, x.shape[3], t, nt),
                 stride=new_stride,
                 storage_offset=stride[1] + nt * stride[-2]).add_(y)

def stripe_parent_right(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, x.shape[3], nt, t),
                            stride=new_stride,
                            storage_offset=w*stride[1] + nt*stride[-1])

def stripe_parent_right_add(x, y, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], n, w, x.shape[3], nt, t),
                     stride=new_stride,
                     storage_offset=w*stride[1] + nt*stride[-1]).add_(y)


# #
#
#
# def stripe_parent_left(x, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[2])
#     new_stride.extend(stride[3:])
#     return x.as_strided(size=(x.shape[0], n,  w,  *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset=stride[2])
#
# def stripe_parent_left_add(x, y, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[2])
#     new_stride.extend(stride[3:])
#     x.as_strided(size=(x.shape[0], n,  w,  *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset=stride[2]).add_(y)
#
#
# def stripe_headed_left(x, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[1])
#     new_stride.extend(stride[3:])
#     return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset= w * stride[2])
#
# def stripe_headed_left_add(x, y, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[1])
#     new_stride.extend(stride[3:])
#     x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset= w * stride[2]).add_(y)
#
#
# def stripe_headed_right(x, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[1])
#     new_stride.extend(stride[3:])
#     return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset= stride[1])
#
# def stripe_headed_right_add(x, y, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[1])
#     new_stride.extend(stride[3:])
#     x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset= stride[1]).add_(y)
#
#
#
# def stripe_parent_right(x, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[2])
#     new_stride.extend(stride[3:])
#     return x.as_strided(size=(x.shape[0], n, w,  *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset=w*stride[1])
#
# def stripe_parent_right_add(x, y, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[2])
#     new_stride.extend(stride[3:])
#     x.as_strided(size=(x.shape[0], n, w,  *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset=w*stride[1]).add_(y)











def stripe_version7(x, n, w=0):
    pass

def stripe_version6(x, n, w=0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2])
    new_stride.append(stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, w, w,  *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=0)



def decode_stripe1(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2]+stride[3])
    new_stride.append(stride[3])
    # new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w),
                            stride=new_stride,
                            storage_offset= (w) * (stride[2]))

def decode_stripe2(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2]+stride[3])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= w * stride[1])

def stripe_logadd_outside(x, y, n, w, offset):
    x = x.contiguous()
    stride =  list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3] )
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3])\
        .copy_(torch.logaddexp(tmp, y))

def stripe_add_outside(x, y, n, w, offset):
    x = x.contiguous()
    stride =  list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    # tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3] )
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3])\
        .add_(y)

def stripe_need_dad_add(x, y,  n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                       storage_offset= start * stride[1] + (end) * stride[2] + headstart * stride[3]).add_(y)



def stripe_need_dad(x, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                       storage_offset= start * stride[1] + (end) * stride[2] + headstart * stride[3])



def stripe_add_outside_v2(x, y, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    # tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
    #                    storage_offset= start * stride[1] + (end) * stride[2] + headstart * stride[3])
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                 storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3]) \
        .add_(y)

def stripe_add_outside_left(x, y, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], n, w-1, *list(x.shape[4:])), stride=new_stride,
                 storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3]) \
        .add_(y)

def stripe_add_outside_right(x, y, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], n, w-1, *list(x.shape[4:])), stride=new_stride,
                 storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3]) \
        .add_(y)




def stripe_logadd_outside_v2(x, y, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                       storage_offset= start * stride[1] + (end) * stride[2] + headstart * stride[3])
    # print(tmp.shape)
    # print(y.shape)
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                 storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3]) \
        .copy_(torch.logaddexp(tmp, y))


def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    else:
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def stripe_outside(x, y, n, w, offset):
    x = x.contiguous()
    stride =  list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3] )
    return tmp


def stripe_copy_gradient_left(x, y, n, left, right):
    x = x.contiguous()
    stride = list(x.stride())
    newstride = []
    newstride.append(stride[0])
    newstride.append(stride[1]+stride[2])
    # newstride.append(stride[1]+stride[2])
    newstride.append(stride[2])
    newstride.extend(stride[2:])
    tmp = x.as_strided(size=(x.shape[0], n, right, left, *x.shape[3:]), stride=newstride,
                      storage_offset=left*stride[2])
    print(tmp.shape)
    print(y.shape)
    x.as_strided(size=(x.shape[0], n, right, left, *x.shape[3:]), stride=newstride,
                storage_offset=right * stride[1]).copy_(torch.logaddexp(tmp, y))


def stripe_copy_gradient_right(x, y, n, left, right):
    x = x.contiguous()
    stride = list(x.stride())
    newstride = []
    newstride.append(stride[0])
    newstride.append(stride[1]+stride[2])
    # newstride.append(stride[1]+stride[2])
    newstride.append(stride[2])
    newstride.extend(stride[2:])
    tmp = x.as_strided(size=(x.shape[0], n, left, right, *x.shape[3:]), stride=newstride,
                      storage_offset=left*stride[2])
    # print(tmp.shape)
    # print(y.shape)
    x.as_strided(size=(x.shape[0], n, left, right, *x.shape[3:]), stride=newstride,
                storage_offset= right * stride[2]).copy_(torch.logaddexp(tmp, y))



# used in lexicalized-pcfg.
def stripe_version2(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(x.shape[0], n, w, w+1, *list(x.shape[4:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)

def stripe_version2_add(x, y, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3]
    stride[2] = (1 if dim == 1 else seq_len) * numel

    x.as_strided(size=(x.shape[0], n, w, w+1, *list(x.shape[4:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel).add_(y)


def stripe_version2_left(x, n, w):
    x = x.contiguous()
    stride = list(x.stride())
    # numel = stride[2]
    new_stride = list(x.stride())
    new_stride[1] = stride[1] + stride[2] + stride[3]
    new_stride[2] = stride[1]
    return x.as_strided(size=(x.shape[0], n-1, w, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= stride[1] +  (w+1) * stride[2] + stride[3])

# s_need_dad, n, w,
def stripe_copy_left(x, y, n, w):
    x = x.contiguous()
    stride = list(x.stride())
    # numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[1])
    new_stride.extend(x.shape[4:])
    x.as_strided(size=(x.shape[0], n-1, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= stride[1] +  (w+1) * stride[2] + stride[3]).copy_(y)

def stripe_version2_right(x, n, w):
    x = x.contiguous()
    stride = list(x.stride())
    # numel = stride[2]
    new_stride = list(x.stride())
    new_stride[1] = stride[1] + stride[2] + stride[3]
    new_stride[2] = stride[2]
    return x.as_strided(size=(x.shape[0], n-1, w, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset=   stride[2])

def stripe_copy_right(x, y, n, w):
    x = x.contiguous()
    stride = list(x.stride())
    # numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[1])
    new_stride.extend(x.shape[4:])
    x.as_strided(size=(x.shape[0], n-1, w,  *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= stride[2]).copy_(y)


def stripe_version_nt_nt(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)

    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    origin_stride = x.stride()
    return x.as_strided(size=(x.shape[0], n, w-2, w-1, *list(x.shape[4:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel + \
                                          2 * (1-dim)*(origin_stride[3])
                                           + (1 - dim) * origin_stride[1] +  dim * origin_stride[2])


def stripe_version_nt_t(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2]+stride[3])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset=  w * numel)

def stripe_version_t_nt(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2]+stride[3])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= stride[1] + (w+1) * stride[2] + stride[3])








# used in lexicalized-pcfg.
def stripe_version4(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3] + stride[4]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(x.shape[0], n, w, w+1, w+1, *list(x.shape[5:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)

# used for arc_indictor; calculating the marginal arc probabilities.
def stripe_arc_indicator(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2])
    new_stride.append(stride[1])
    new_stride.append(stride[2])
    return x.as_strided(size=(x.shape[0], n, w, w),
                            stride=new_stride,
                            storage_offset=0)



#for lexicalized_pcfg.
def diagonal_copy_v2(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).copy_(y)
    else:
        x.as_strided(size=(x.shape[0], seq_len - w, w),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).copy_(y)

def diagonal_copy_v2_add(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).add_(y)
    else:
        new_stride.append(stride[3])
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).add_(y)


def diagonal2(x,  w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        return x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     )
    else:
        new_stride.append(stride[3])
        return x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     )


#for lexicalized_pcfg.
def diagonal_copy_logadd_v2(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        tmp = x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     )
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(torch.logaddexp(tmp, y))

    else:
        raise NotImplementedError
        new_stride.append(stride[3])
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).copy_(y)


def diagonal_v2(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        return x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     )
    else:
        raise NotImplementedError




# def stripe()




def diagonal_copy_v4(x, y, nth_diagonal, total_num,  width, start_offset=0, head_offset=0, head_moving=0):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3] + head_moving * stride[4])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    size = (x.shape[0], total_num, width, *list(y.shape[3:]))
    x.as_strided(size=size,
                 stride=new_stride,
                 storage_offset= start_offset * stride[1] + (start_offset + nth_diagonal) * (stride[2])   + (head_offset) * stride[4]
                 ).copy_(y)

def diagonal_copy_v3(x, y, nth_diagonal, total_num,  width, start_offset=0, head_offset=0, head_moving=1):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + head_moving*stride[3])
    new_stride.append(stride[3])

    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        size = (x.shape[0], total_num, width, *list(x.shape[4:]))
        x.as_strided(size=size,
                     stride=new_stride,
                     storage_offset= start_offset * stride[1] + (start_offset + nth_diagonal) * stride[2] + (head_offset) * stride[3]
                     ).copy_(y)
    else:
        raise NotImplemented

def diagonal_copy_(x, y, w):
    # size of x: (batch, N, N, nt)
    # size of y: (batch, N, nt)
    # the function aims to copy y to the diagonal of x (dim1 and dim2) without any copy of tensor.
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(size=(x.shape[0], seq_len - w,  *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).copy_(y)
    else:
        x.as_strided(size=(x.shape[0], seq_len - w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(y)




def diagonal(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )
    else:
        return x.as_strided(size=(x.shape[0], seq_len - w),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )









