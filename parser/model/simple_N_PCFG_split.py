import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from parser.pcfgs.simple_split_pcfg import SimplePCFG_Triton

class Simple_N_PCFG_split(nn.Module):
    def __init__(self, args, dataset):
        super(Simple_N_PCFG_split, self).__init__()
        self.pcfg = SimplePCFG_Triton()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.NT

        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        # self.r = args.r_dim
        # self.word_emb_size = args.word_emb_size
        rule_dim = self.s_dim

        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        #terms
        # self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
        )

        self.vocab_emb =  nn.Parameter(torch.randn(self.s_dim, self.V))
        
                                    #   nn.Linear(self.s_dim, self.V))
        self.rule_state_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        # self.parent_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU())
        # self.root_mlp =  nn.Sequential(nn.Linear(rule_dim,rule_dim),                                       ResLayer(self.s_dim, self.s_dim),
        #                               ResLayer(self.s_dim, self.s_dim),)
        
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),
                                      nn.ReLU()
                                      )
         
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),
                                       nn.ReLU()
                                       )
        self.parent_mlp1 =  nn.Sequential(nn.Linear(self.s_dim, self.s_dim),              nn.ReLU(),
                                      )
        

        self.split = nn.Sequential(
            nn.Linear(rule_dim, 2))
    
        # self.rank_proj = nn.Parameter(torch.randn(rule_dim, self.r))
        self._initialize()  


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input, **kwargs):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            roots = (self.root_emb  @ self.rule_state_emb.t())
            roots = roots.log_softmax(-1)
            return roots.expand(b, roots.shape[-1])

        def terms():
            term_emb = self.rule_state_emb
            term_prob = ((self.term_mlp(term_emb) + term_emb) @ self.vocab_emb).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None,None], x[:, :, None]]
        
            # term_prob = term_prob.unsqueeze(0).unsqueeze(1).expand(
            #     b, n, self.T, self.V
            # )
            # indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            # term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            # return term_prob

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb
            parent1 = self.parent_mlp1(nonterm_emb) + nonterm_emb   

            # parent2 = self.parent_mlp2(nonterm_emb) + nonterm_emb   
            left = (self.left_mlp(rule_state_emb) + rule_state_emb) @  parent1.t()
            right = (self.right_mlp(rule_state_emb) + rule_state_emb) @ parent1.t()
            # right = left

            # head = head.softmax(-1)
            left = left.softmax(-2)
            right = right.softmax(-2)

            # left_m =  left[:self.NT, :]
            # left_p =  left[self.NT:, :]
            
            # right_m = right[:self.NT, :]
            # right_p = right[self.NT:, :]            

            return (left, right)

        root, unary, (left_m,  right_m) = roots(), terms(), rules()

        split = self.split(self.rule_state_emb).log_softmax(-1)

        return {'unary': unary,
                'root': root,
                # 'head': head,
                'left_m': left_m,
                'right_m': right_m,
                # 'left_p': left_p,
                # 'right_p' : right_p,
                'split': split,
                'kl': 0}


    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        logZ =  -result['partition'].mean()
        # hmm_loss = -self.pcfg._hmm_forward(rules=rules, lens=input['seq_len']).mean()
        return logZ 


    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == 'viterbi':
            assert NotImplementedError

        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError