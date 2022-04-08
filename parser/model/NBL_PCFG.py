import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from ..pcfgs.blpcfg import BLPCFG, FastBLPCFG
from ..pcfgs.eisner_satta import EisnerSatta


class NeuralBLPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralBLPCFG, self).__init__()
        self.pcfg = BLPCFG()

        self.device = dataset.device
        self.args = args
        # number of states
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.NT_T = self.NT + self.T
        self.dataset = dataset

        # embedding dimensions
        self.s_dim = args.s_dim
        self.r = args.r

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.nonterm_emb_root = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.noninherent_emb = nn.Parameter(torch.randn(self.NT_T, self.s_dim))
        self.inherent_emb = nn.Parameter(torch.randn(self.NT_T, self.s_dim))

        self.word_emb = nn.Embedding(self.V, self.s_dim)

        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.r_emb = nn.Parameter(torch.randn(self.r, self.s_dim))
        self.r_emb2 = nn.Parameter(torch.randn(self.r, self.s_dim))
        self.r_emb3 = nn.Parameter(torch.randn(self.r, self.s_dim))

        self.beta_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V)
                                      )

        self.noninherent_mlp = nn.Sequential(
                                             nn.Linear(self.s_dim, self.NT_T * 2))

        self.inherent_mlp = nn.Sequential(nn.Linear(self.s_dim, self.NT_T))


        self.head_encoder = nn.Linear(self.s_dim + self.s_dim, self.s_dim)

        self.head_mlp = nn.Sequential(
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.r)
        )

        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT))

        self.root_mlp2 = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                       ResLayer(self.s_dim, self.s_dim),
                                       ResLayer(self.s_dim, self.s_dim),
                                       nn.Linear(self.s_dim, self.V))

        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input):
        x = input['word']
        b, n = x.shape[:2]
        x_emb = self.word_emb(x)

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb).log_softmax(-1)
            roots_v = self.root_mlp2(self.nonterm_emb).log_softmax(-1)
            roots_v = torch.gather(roots_v.unsqueeze(0).expand(b, self.NT, self.V), -1,
                                   x.unsqueeze(1).expand(-1, self.NT, -1))
            return roots.expand(b, self.NT).unsqueeze(1) + roots_v.transpose(-1, -2)

        nt_emb = self.nonterm_emb.unsqueeze(0).expand(b, -1, -1)
        x_emb = x_emb.unsqueeze(2).expand(-1, -1, self.NT, -1)
        nt_x_emb = torch.cat([x_emb,
                              nt_emb.unsqueeze(1).expand(-1, n, -1, -1)], dim=3
                             )

        head = self.head_mlp(self.head_encoder(nt_x_emb).relu() + x_emb)

        head = head.log_softmax(-1)

        inherent = self.inherent_mlp(self.r_emb).log_softmax(-1).permute(1, 0).unsqueeze(0).expand(b, -1, -1)

        noninherent_symbol = self.noninherent_mlp(self.r_emb2).log_softmax(-1).reshape(self.r, self.NT_T, 2).transpose(1, 0).unsqueeze(0).expand(b, -1,
                                                                                                            -1, -1)

        noninherent_word = self.beta_mlp(self.r_emb3).log_softmax(-1).transpose(1, 0).unsqueeze(0).expand(b, -1, -1)
        noninherent_word = torch.gather(noninherent_word, 1, x.unsqueeze(-1).expand(-1, -1, self.r))
        noninherent = noninherent_word.unsqueeze(-2).unsqueeze(-1) + noninherent_symbol.unsqueeze(1)

        return {
            'head': head,
            'noninherent': noninherent,
            'inherent': inherent,
            'root': roots(),
            'kl': 0}



    def forward4viterbi(self, input):
        x = input['word']
        b, n = x.shape[:2]
        x_emb = self.word_emb(x)

        nt_emb = self.nonterm_emb.unsqueeze(0).expand(b, -1, -1)
        x_emb = x_emb.unsqueeze(2).expand(-1, -1, self.NT, -1)
        nt_x_emb = torch.cat([x_emb,
                              nt_emb.unsqueeze(1).expand(-1, n, -1, -1)], dim=3
                             )

        head = self.head_mlp(self.head_encoder(nt_x_emb).relu() + x_emb)
        head = head.softmax(-1)

        inherent = self.inherent_mlp(self.r_emb).softmax(-1).permute(1, 0).unsqueeze(0).expand(b, -1, -1)

        noninherent_symbol = self.noninherent_mlp(self.r_emb2).softmax(-1).reshape(self.r, self.NT_T, 2).transpose(1,
                                                                                                0).unsqueeze(0).expand(b, -1, -1, -1)

        noninherent_word = self.beta_mlp(self.r_emb3).softmax(-1).transpose(1, 0).unsqueeze(
                0).expand(b, -1, -1)

        noninherent_word = torch.gather(noninherent_word, 1, x.unsqueeze(-1).expand(-1, -1, self.r))

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb).log_softmax(-1)
            roots_v = self.root_mlp2(self.nonterm_emb).log_softmax(-1)
            roots_v = torch.gather(roots_v.unsqueeze(0).expand(b, self.NT, self.V), -1,
                                   x.unsqueeze(1).expand(-1, self.NT, -1))
            return roots.expand(b, self.NT).unsqueeze(1) + roots_v.transpose(-1, -2)

        def rule():
            rule = torch.zeros(b, n, n, self.NT, self.NT_T, self.NT_T,2, device=torch.device('cuda'))
            #### to avoid cuda out-of-memory:
            for i in range(self.NT):
                rule[:, :, :, i, ...] = torch.einsum("qnr, qbrd, qmr, qcr -> qmnbcd", head[:, :, i], noninherent_symbol,noninherent_word,  inherent).add_(1e-15).log_()
            left = rule[..., 1].transpose(-1, -2)
            right = rule[..., 0]
            rule = left.mul_(torch.tril(torch.ones(n, n, device=torch.device("cuda")), diagonal=-1).unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1).expand(*left.shape))\
                   + right.mul_(torch.triu(torch.ones(n, n, device=torch.device("cuda")), diagonal=1).unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1).expand(*right.shape))
            return rule

        return {
            'root': roots(),
            'rule': rule(),
            'kl': 0
            }





    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg.loss(rules, input['seq_len'])
        return -result['partition'].mean()

    def evaluate(self, input, decode_type='mbr', eval_dep=False):
        if decode_type == 'mbr':
            rules = self.forward(input)
            return self.pcfg.decode(rules, input['seq_len'], mbr=True, eval_dep=eval_dep)
        else:
            rules = self.forward4viterbi(input)
            result = EisnerSatta.viterbi_decoding(rule=rules['rule'], root=rules['root'],lens=input['seq_len'])
            rules = self.forward(input)
            logZ = self.pcfg.loss(rules, input['seq_len'])
            result.update(logZ)
            return result




class FastNBLPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(FastNBLPCFG, self).__init__()
        self.pcfg = FastBLPCFG()

        self.device = dataset.device
        self.args = args
        # number of states
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.NT_T = self.NT + self.T
        self.dataset = dataset

        # embedding dimensions
        self.s_dim = args.s_dim
        self.r = args.r

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.nonterm_emb_root = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.noninherent_emb = nn.Parameter(torch.randn(self.NT_T, self.s_dim))
        self.inherent_emb = nn.Parameter(torch.randn(self.NT_T, self.s_dim))

        self.word_emb = nn.Embedding(self.V, self.s_dim)

        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.r_emb = nn.Parameter(torch.randn(self.r, self.s_dim))
        self.r_emb2 = nn.Parameter(torch.randn(self.r, self.s_dim))
        self.r_emb3 = nn.Parameter(torch.randn(self.r, self.s_dim))

        self.beta_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V)
                                      )

        self.noninherent_mlp = nn.Sequential(
                                             nn.Linear(self.s_dim, self.NT_T * 2))

        self.inherent_mlp = nn.Sequential(nn.Linear(self.s_dim, self.NT_T))
        self.head_encoder = nn.Linear(self.s_dim + self.s_dim, self.s_dim)

        self.head_mlp = nn.Sequential(
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.r)
        )

        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT))

        self.root_mlp2 = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                       ResLayer(self.s_dim, self.s_dim),
                                       ResLayer(self.s_dim, self.s_dim),
                                       nn.Linear(self.s_dim, self.V))
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input):
        x = input['word']
        b, n = x.shape[:2]
        x_emb = self.word_emb(x)

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb).log_softmax(-1)
            roots_v = self.root_mlp2(self.nonterm_emb).log_softmax(-1)
            roots_v = torch.gather(roots_v.unsqueeze(0).expand(b, self.NT, self.V), -1,
                                   x.unsqueeze(1).expand(-1, self.NT, -1))
            return roots.expand(b, self.NT).unsqueeze(1) + roots_v.transpose(-1, -2)

        nt_emb = self.nonterm_emb.unsqueeze(0).expand(b, -1, -1)
        x_emb = x_emb.unsqueeze(2).expand(-1, -1, self.NT, -1)
        nt_x_emb = torch.cat([x_emb,
                              nt_emb.unsqueeze(1).expand(-1, n, -1, -1)], dim=3
                             )
        head = self.head_mlp(self.head_encoder(nt_x_emb).relu() + x_emb)
        head = head.softmax(-1)
        inherent = self.inherent_mlp(self.r_emb).softmax(-1).permute(1, 0).unsqueeze(0).expand(b, -1, -1)
        noninherent_symbol = self.noninherent_mlp(self.r_emb2).softmax(-1).reshape(self.r, self.NT_T, 2).transpose(1, 0).unsqueeze(0).expand(b, -1,
                                                                                                            -1, -1)
        noninherent_word = self.beta_mlp(self.r_emb3).log_softmax(-1).transpose(1, 0).unsqueeze(0).expand(b, -1, -1)
        noninherent_word = torch.gather(noninherent_word, 1, x.unsqueeze(-1).expand(-1, -1, self.r))

        # b, n, nt, r, 2
        return {
            'head': head,
            'noninherent_symbol': noninherent_symbol,
            'noninherent_word': noninherent_word,
            'inherent': inherent,
            'root': roots(),
            'kl': 0}

    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg.loss(rules, input['seq_len'])
        return -result['partition'].mean()

    def evaluate(self, input, decode_type='mbr', eval_dep=False):
        if decode_type == 'mbr':
            rules = self.forward(input)
            return self.pcfg.decode(rules, input['seq_len'], mbr=True, eval_dep=eval_dep)
        else:
            rules = self.forward4viterbi(input)
            result = EisnerSatta.viterbi_decoding(rule=rules['rule'], root=rules['root'],lens=input['seq_len'])
            rules = self.forward(input)
            logZ = self.pcfg.loss(rules, input['seq_len'])
            result.update(logZ)
            return result


