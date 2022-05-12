import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from ..pcfgs.tdpcfg import TDPCFG, Fastest_TDPCFG

class TNPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(TNPCFG, self).__init__()
        self.pcfg = TDPCFG()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        self.r = args.r_dim
        self.word_emb_size = args.word_emb_size

        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        #terms
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT+self.T, self.s_dim))
        rule_dim = self.s_dim
        self.parent_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU(),nn.Linear(rule_dim,self.r))
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim), nn.ReLU(),nn.Linear(rule_dim,self.r))
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU(),nn.Linear(rule_dim,self.r))


    def forward(self, input, **kwargs):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            roots = self.root_mlp(self.root_emb).log_softmax(-1)
            return roots.expand(b, roots.shape[-1]).contiguous()

        def terms():
            term_prob = self.term_mlp(self.term_emb).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None,None], x[:, :, None]]

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[:self.NT]
            head = self.parent_mlp(nonterm_emb).log_softmax(-1)
            left = self.left_mlp(rule_state_emb).log_softmax(-2)
            right = self.right_mlp(rule_state_emb).log_softmax(-2)
            head = head.unsqueeze(0).expand(b,*head.shape)
            left = left.unsqueeze(0).expand(b,*left.shape)
            right = right.unsqueeze(0).expand(b,*right.shape)
            return (head, left, right)

        root, unary, (head, left, right) = roots(), terms(), rules()

        return {'unary': unary,
                'root': root,
                'head': head,
                'left': left,
                'right': right,
                'kl': 0}

    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        logZ =  -result['partition'].mean()
        return logZ


    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == 'viterbi':
            assert NotImplementedError

        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError



class FastTNPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(FastTNPCFG, self).__init__()
        self.pcfg = Fastest_TDPCFG()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        self.r = args.r_dim
        self.word_emb_size = args.word_emb_size

        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      # )
                                      nn.Linear(self.s_dim, self.NT))

        #terms
        # self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT+self.T, self.s_dim))
        rule_dim = self.s_dim
        self.parent_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU())
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU())
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU())

        self.rank_proj = nn.Parameter(torch.randn(rule_dim, self.r))


        self._initialize()


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def forward(self, input, **kwargs):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            roots = self.root_mlp(self.root_emb)
            roots = roots.log_softmax(-1)
            return roots.expand(b, roots.shape[-1])

        def terms():
            term_emb = self.rule_state_emb[self.NT:]
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None,None], x[:, :, None]]
            # term_prob = term_prob.unsqueeze(0).unsqueeze(1).expand(
            #     b, n, self.T, self.V
            # )
            # indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            # term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            # return term_prob

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[:self.NT]
            head = self.parent_mlp(nonterm_emb) @ self.rank_proj
            left = self.left_mlp(rule_state_emb) @ self.rank_proj
            right = self.right_mlp(rule_state_emb) @ self.rank_proj
            head = head.softmax(-1)
            left = left.softmax(-2)
            right = right.softmax(-2)
            head = head.unsqueeze(0).expand(b,*head.shape)
            left = left.unsqueeze(0).expand(b,*left.shape)
            right = right.unsqueeze(0).expand(b,*right.shape)
            return (head, left, right)

        root, unary, (head, left, right) = roots(), terms(), rules()

        return {'unary': unary,
                'root': root,
                'head': head,
                'left': left,
                'right': right,
                'kl': 0}

    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        logZ =  -result['partition'].mean()
        return logZ


    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == 'viterbi':
            assert NotImplementedError

        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError
