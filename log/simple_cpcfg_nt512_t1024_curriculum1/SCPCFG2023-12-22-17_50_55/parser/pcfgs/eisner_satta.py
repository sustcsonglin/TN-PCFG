from .pcfgs import PCFG_base
import torch
from .fn import *

'''
O(l^4m^2+l^3m^3) inference with Bilexicalized PCFGs using (Eisner and Satta, 1999) 
'''
class EisnerSatta():

    @classmethod
    def viterbi_decoding(cls, rule, root, lens):
        B, H, _, NT, S, _ = rule.shape
        T = S - NT
        N = H + 1
        nt_slice = slice(0, NT)
        t_slice = slice(NT, S)
        s = rule.new_zeros(B, N, N, H, NT).fill_(-1e9)

        # for back-tracking.
        s_bp = rule.new_zeros(B, N, N, H, NT).fill_(-1e9)

        s_need_dad = rule.new_zeros(B, N, N, H, NT, NT).fill_(-1e9)

        # if we want to know the state of t, then we can use this tensor to look up.
        s_inherent = rule.new_zeros(B, N, N, NT, NT).fill_(-1e9)

        for i in range(N - 1):
            s[:, i, i + 1, i] = 0
            if i > 0:
                s_need_dad[:, i, i + 1, :i], s_need_dad[:, i + 1, i, :i] = (rule[:, i, :i, :, nt_slice, t_slice]).max(-1)
            if i < N - 2:
                s_need_dad[:, i, i + 1, i + 1:], s_need_dad[:, i + 1, i, i + 1:] = (rule[:, i, i + 1:, :, t_slice, nt_slice]).max(-2)
        LEFT = 0
        RIGHT = 1
        for w in range(2, N):
            n = N - w
            if w == 2:
                headed = s.new_zeros(B, n, w, NT, T, T).fill_(-1e9)
                headed[:, :, LEFT, ...] = rule[:, torch.arange(n) + 1, torch.arange(n), :, t_slice, t_slice]
                headed[:, :, RIGHT, ...] = rule[:, torch.arange(n), torch.arange(n) + 1, :, t_slice, t_slice]
                headed, idx = headed.reshape(B, n, w, NT, -1).max(-1)
                diagonal_copy_with_headword(s, headed, w)
                diagonal_copy_with_headword(s_bp, idx, w)
            else:
                left_span = stripe_with_headword(s, n, w - 1, (0, 1))
                right_span = stripe_with_headword(s, n, w - 1, (1, w), 0)
                headed = s.new_zeros(B, n, w - 1, w, NT, NT).fill_(-1e9)

                for i in range(w - 1):
                    if i > 0:
                        need_left_dad = stripe_need_dad(s_need_dad, n, i + 1, start=i + 1, end=w, headstart=0)
                        headed[:, :, i, :i + 1] = (left_span[:, :, i, :i + 1, ..., None, :] + need_left_dad)

                    else:
                        a, b = (right_span[:, :, 0, 1:, ..., None, None, :] +
                             stripe_headed_right(rule, n, w - 1, NT, T)).permute(0,1,3,5,2,4).reshape(B, n, NT, NT, -1).max(-1)
                        s_inherent[:, torch.arange(n), torch.arange(n) + w] = b.float()
                        headed[:, :, i, 0] = a

                    if i < w - 2:
                        need_right_dad = stripe_need_dad(s_need_dad, n, w - (i + 1), start=0, end=i + 1,
                                                         headstart=i + 1)
                        headed[:, :, i, i + 1:] = (right_span[:, :, i, i + 1:, ..., None, :] + need_right_dad)

                    else:
                        a, b = (left_span[:, :, w - 2, :w - 1, ..., None, :, None] +
                                stripe_headed_left(rule, n, w - 1, NT, T)).permute(0, 1, 3, 4, 2, 5).reshape(B, n, NT, NT, -1).max(-1)
                        headed[:, :, i, -1] = a
                        s_inherent[:, w + torch.arange(n), torch.arange(n)] = b.float()

                headed, idxes = headed.permute(0, 1, 3, 4, 2, 5).reshape(B, n, w, NT, -1).max(-1)
                diagonal_copy_with_headword(s, headed, w)
                diagonal_copy_with_headword(s_bp, idxes, w)

            if w < N - 1:
                for l in range(N - w):
                    r = w + l
                    if l > 0:
                        u = (rule[:, l:r, :l, :, nt_slice, nt_slice]
                             + headed[:, l, :, None, None, None,:]).permute(0, 2, 3, 4, 1, 5)
                        maxes, idx = u.reshape(*u.shape[:-2], -1).max(-1)
                        s_need_dad[:, l, r, :l] = maxes
                        s_need_dad[:, r, l, :l] = idx
                    if r < N - 1:
                        u = (rule[:, l:r, r:, :, nt_slice, nt_slice]
                             + headed[:, l, :, None, None, :,None]).permute(0, 2, 3, 5, 1,4)
                        maxes, idx = u.reshape(*u.shape[:-2], -1).max(-1)
                        s_need_dad[:, l, r, r:] = maxes
                        s_need_dad[:, r, l, r:] = idx

        maxes, states = (s[torch.arange(B), 0, lens, :, :] + root).max(-1)
        maxes, head_idx = maxes.max(-1)

        def backtrack(b, left, right, head, head_state):
            assert left < right
            assert head < right
            assert head >= left
            if right == left + 1:
                if head == left:
                    return [(left, right, head_state, head)], []
                else:
                    return [(left, right, head_state, head)], []

            if right == left + 2:
                child_states = int(s_bp[b, left, right, head, head_state])
                left_state = int(child_states / T) + NT
                right_state = child_states % T + NT
                lspan, _ = backtrack(b, left, left + 1, left, left_state)
                rspan, _ = backtrack(b, left + 1, right, left + 1, right_state)
                return [(left, right, head_state, head)] + lspan + rspan, [
                    (left, right - 1) if head == left else (head, left)]

            inherent = int(s_bp[b, left, right, head, head_state])
            inherent_state = inherent % NT
            split = int(inherent / NT)
            inherent_head = head
            split += left + 1

            ## means that inherent state (left) belongs to T.
            if (split == left + 1) and (head == left):
                i_t = int(s_inherent[b, left, right, head_state, inherent_state])
                ni_split = int(i_t / T)
                t_states = i_t % T
                noninherent_state = inherent_state
                inherent_state = t_states
                noninherent_head = split + ni_split
                lspan, larc = backtrack(b, left, split, left, inherent_state)
                rspan, rarc = backtrack(b, split, right, noninherent_head, noninherent_state)
                return [(left, right, head_state, head)] + lspan + rspan, [(head, noninherent_head)] + larc + rarc

            ## means that inherent state (right) belongs to T.
            if (split == right - 1) and (head == right - 1):
                i_t = int(s_inherent[b, right, left, head_state, inherent_state])
                ni_split = int(i_t / T)
                t_states = i_t % T
                noninherent_state = inherent_state
                inherent_state = t_states
                noninherent_head = left + ni_split
                lspan, larc = backtrack(b, left, split, noninherent_head, noninherent_state)
                rspan, rarc = backtrack(b, split, right, right - 1, inherent_state)
                return [(left, right, head_state, head)] + lspan + rspan, [(head, noninherent_head)] + larc + rarc

            if split > head:
                direction = 0
            else:
                direction = 1

            if direction == 0:
                noninherent = int((s_need_dad[b, right, split, head, head_state, inherent_state]))
                if split != right - 1:
                    noninherent_state = noninherent % NT
                    noninherent_split = int(noninherent / NT)
                    noninherent_head = (split) + noninherent_split
                else:
                    noninherent_head = right - 1
                    noninherent_state = noninherent + NT
                lspan, larc = backtrack(b, left, split, inherent_head, inherent_state)
                rspan, rarc = backtrack(b, split, right, noninherent_head, noninherent_state)

            else:
                noninherent = int((s_need_dad[b, split, left, head, head_state, inherent_state]))
                if split != left + 1:
                    noninherent_state = noninherent % NT
                    noninherent_split = int(noninherent / NT)
                    noninherent_head = (left) + noninherent_split
                else:
                    noninherent_state = noninherent + NT
                    noninherent_head = left
                lspan, larc = backtrack(b, left, split, noninherent_head, noninherent_state)
                rspan, rarc = backtrack(b, split, right, inherent_head, inherent_state)
            return [(left, right, head_state, head)] + lspan + rspan, [(head, noninherent_head)] + larc + rarc

        predict_span = []
        predict_arc = []

        for b in range(B):
            p_s, p_a = backtrack(b, 0, N - 1, int(head_idx[b]), int((states[b][head_idx[b]])))
            predict_span.append(p_s)
            predict_arc.append(p_a + [(-1, int(head_idx[b]))])

        del s, s_bp, s_need_dad
        return {
            'prediction':predict_span,
            'prediction_arc':predict_arc,
            'partition': 0
        }



