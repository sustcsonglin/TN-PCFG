import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.checkpoint import checkpoint as ckp
from parser.pcfgs.simple_pcfg import SimplePCFG_Triton_Batch

'''
To save memory of the inside algorithm.
'''


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return ckp(func, *args, **kwargs)

    return wrapper

class Simple_C_PCFG(nn.Module):
    def __init__(self, args, dataset):
        super(Simple_C_PCFG, self).__init__()
        self.pcfg = SimplePCFG_Triton_Batch()

        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        self.z_dim = args.z_dim
        self.enc_dim = args.h_dim

        # self.r = args.r_dim
        self.word_emb_size = args.w_dim
        rule_dim = self.s_dim


        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        input_dim = self.s_dim + self.z_dim

        #terms
        # self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.term_mlp = nn.Sequential(nn.Linear(input_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V)
        )

        # self.vocab_emb =  nn.Parameter(torch.randn(self.s_dim, self.V))
                                    #   nn.Linear(self.s_dim, self.V))

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT+self.T, self.s_dim))
        # self.parent_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU())
        # self.root_mlp =  nn.Sequential(nn.Linear(rule_dim,rule_dim),                                       ResLayer(self.s_dim, self.s_dim),
        #                               ResLayer(self.s_dim, self.s_dim),)
        
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU()) 
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU())
        self.parent_mlp1 =  nn.Sequential(nn.Linear(input_dim, self.s_dim),
                                          nn.ReLU(),
                                      )
    
        self.enc_emb = nn.Embedding(self.V, 512)
        self.enc_rnn = nn.LSTM(512, 512, bidirectional=True, num_layers=1, batch_first=True)
        self.enc_out = nn.Linear(512 * 2, self.z_dim * 2)



        # self.rank_proj = nn.Parameter(torch.randn(rule_dim, self.r))
        self._initialize()  

    

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input, evaluating=False, **kwargs):
        x = input['word']
        b, n = x.shape[:2]
        seq_len = input['seq_len']

        def enc(x):
            x_embbed = self.enc_emb(x)
            x_packed = pack_padded_sequence(
                x_embbed, seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.enc_rnn(x_packed)
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
            out = self.enc_out(h)
            mean = out[:, : self.z_dim]
            lvar = out[:, self.z_dim :]
            return mean, lvar

        def kl(mean, logvar):
            result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
            return result

        mean, lvar = enc(x)
        z = mean

        if not evaluating:
            z = mean.new(b, mean.size(1)).normal_(0,1)
            z = (0.5 * lvar).exp() * z + mean

        def roots():
            roots = (self.root_emb @ self.rule_state_emb[:self.NT].t())
            roots = roots.log_softmax(-1)
            return roots.expand(b, roots.shape[-1])

        def terms():
            term_emb = self.rule_state_emb[self.NT:].unsqueeze(0).expand(
                b, self.T, self.s_dim
            )            

            # to_stack = []            
            # for i in range(b):
                #  to_stack.append(compute_per_batch(term_emb, z[i], x[i]))            
            # return torch.stack(to_stack, dim=0)
                # b, self.T, self.s_dim
            # )

            z_expand = z.unsqueeze(1).expand(b, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            return term_prob.gather(-1, x.unsqueeze(1).expand(b, self.T, x.shape[-1])).transpose(-1, -2)
        
        # @checkpoint
        # def compute_per_batch(p_emb, z, x):
            # return term_prob[torch.arange(self.T)[None,None], x[:, :, None]]
            # term_prob = self.term_mlp(torch.cat([p_emb, z.unsqueeze(0).expand(p_emb.shape[0], z.shape[-1])], dim=-1)).log_softmax(-1)
        # T = p_emb.shape[0]
        # return  term_prob[torch.arange(T)[None], x[:, None]]


        def rules():
            nonterm_emb = self.rule_state_emb[:self.NT].unsqueeze(0).expand(
                b, self.NT, self.s_dim
            )
            z_expand = z.unsqueeze(1).expand(
                b, self.NT, self.z_dim
            )
            nonterm_emb2 = torch.cat([nonterm_emb, z_expand], -1)

            parent1 = self.parent_mlp1(nonterm_emb2) + nonterm_emb

            # parent2 = self.parent_mlp2(nonterm_emb) + nonterm_emb   
            left = torch.einsum('bnr, mr -> bmn', parent1, (self.left_mlp(self.rule_state_emb) + self.rule_state_emb))
            right =  torch.einsum('bnr, mr -> bmn', parent1, (self.right_mlp(self.rule_state_emb) + self.rule_state_emb))

            # right = left

            # head = head.softmax(-1)
            left = left.softmax(-2)
            right = right.softmax(-2)

            left_m =  left[:, :self.NT, :].contiguous()
            left_p =  left[:, self.NT:, :].contiguous()
            
            right_m = right[:, :self.NT, :].contiguous()
            right_p = right[:, self.NT:, :].contiguous()
            
            return (left_m, left_p, right_m, right_p)

        root, unary, (left_m, left_p, right_m, right_p) = roots(), terms(), rules()

        return {'unary': unary,
                'root': root,
                # 'head': head,
                'left_m': left_m,
                'right_m': right_m,
                'left_p': left_p,
                'right_p' : right_p,
                'kl': kl(mean, lvar).sum(1)}

    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        loss =  (-result['partition'] + rules['kl']).mean()
        return loss



    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == 'viterbi':
            assert NotImplementedError

        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError
        

