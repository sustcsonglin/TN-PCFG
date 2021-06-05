import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from ..pcfgs.lpcfg import L_PCFG
from ..pcfgs.eisner_satta import EisnerSatta
'''
https://github.com/neulab/neural-lpcfg/blob/master/models.py
without compound parameterization
in order to investigate the effect of bilexicalized dependencies.
'''
class NeuralLPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralLPCFG, self).__init__()
        self.pcfg = L_PCFG()
        self.device = dataset.device
        self.args = args
        # number of states
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.NT_T = self.NT + self.T

        # embedding dimensions
        self.s_dim = args.s_dim

        # embeddings
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.nonterm_emission_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.word_emb = nn.Embedding(self.V, self.s_dim)

        self.head_mlp = nn.Sequential(nn.Linear(self.s_dim + self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, 2*self.NT_T))

        self.left_rule_mlp = nn.Linear(self.s_dim + self.s_dim, (self.NT_T) ** 2)
        self.right_rule_mlp = nn.Linear(self.s_dim + self.s_dim, (self.NT_T) ** 2)

        # root rule.
        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        # unary rule
        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)




    def forward(self, input, eval_dep=False, **kwargs):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots.expand(b, self.NT)

        def terms():
            term_emb = self.term_emb
            term_prob = self.term_mlp(torch.cat([self.nonterm_emission_emb,term_emb], dim=0)).log_softmax(-1)
            indices = x.unsqueeze(2).expand(b, n, self.NT + self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob.unsqueeze(0).unsqueeze(0).expand(b, n, *term_prob.shape),
                                     3, indices).squeeze(3)
            return term_prob

        def rules():
            x_emb = self.word_emb(x)
            nt_emb = self.nonterm_emb.unsqueeze(0).expand(b,  -1, -1)
            nt_x_emb = torch.cat([x_emb.unsqueeze(2).expand(-1, -1, self.NT, -1),
                                  nt_emb.unsqueeze(1).expand(-1, n, -1, -1)], dim=3
                                 )
            left_rule_score = self.left_rule_mlp(nt_x_emb)  # nt x t**2
            right_rule_score = self.right_rule_mlp(nt_x_emb)  # nt x t**2
            left_rule_scores = left_rule_score.reshape(b, n, self.NT, self.NT_T, self.NT_T)
            right_rule_scores = right_rule_score.reshape(b,  n, self.NT, self.NT_T, self.NT_T)
            head_score = self.head_mlp(nt_x_emb).log_softmax(-1)  # nt x t**2
            head_scores = head_score.reshape(b, n, self.NT, self.NT_T, 2)
            left_scores = left_rule_scores.log_softmax(dim=-1)
            right_scores = right_rule_scores.log_softmax(dim=-2)
            rule_scores = torch.stack([head_scores[:, :, :, :, 0].unsqueeze(4) + left_scores,
                                 head_scores[:, :, :, :, 1].unsqueeze(3) + right_scores], dim=1)
            return rule_scores

        root, unary, rule = roots(), terms(), rules()


        if eval_dep:
            left = torch.einsum("qnabc, qmc -> qmnabc", rule[:, 0].exp(), unary.exp()).add_(1e-20).log()
            right = torch.einsum("qnabc, qmb -> qmnabc", rule[:, 1].exp(), unary.exp()).add_(1e-20).log()

            rule = left.mul_(
                torch.tril(torch.ones(n, n, device=torch.device("cuda")), diagonal=-1).unsqueeze(0).unsqueeze(
                    -1).unsqueeze(-1).unsqueeze(-1).expand(*left.shape)) \
                   + right.mul_(
                torch.triu(torch.ones(n, n, device=torch.device("cuda")), diagonal=1).unsqueeze(0).unsqueeze(
                    -1).unsqueeze(-1).unsqueeze(-1).expand(*right.shape))
            return {
                'rule': rule,
                'root': root.unsqueeze(1).expand(b, n, self.NT)
            }

        else:
            return {'unary': unary,
                'root': root,
                'rule': rule,
                'kl': torch.tensor(0)}

    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg.loss(rules, input['seq_len'])
        return -result['partition'].mean()

    def evaluate(self, input, decode_type='mbr', eval_dep=False):
        if decode_type == 'mbr':
            rules = self.forward(input)
            return self.pcfg.decode(rules, input['seq_len'], mbr=True, eval_dep=eval_dep)

        else:
            rules = self.forward(input, eval_dep=True)
            result = EisnerSatta.viterbi_decoding(rule=rules['rule'], root=rules['root'],lens=input['seq_len'])
            rules = self.forward(input)
            logZ = self.pcfg.loss(rules, input['seq_len'])
            result.update(logZ)
            return result



