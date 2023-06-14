import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from parser.pcfgs.simple_split_pcfg import SimplePCFG_Triton
from parser.pcfgs.jelinek_lafferty import Jelinek_Lafferty
import pdb

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config





# GPT-2
class Autoregressive_PCFG(nn.Module):
    def __init__(self, args, dataset):
        super(Autoregressive_PCFG, self).__init__()
        print("Hello there...")
        self.pcfg = Jelinek_Lafferty()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.NT

        # self.kl_loss = torch.nn.KLDivLos(reduction="batchmean", log_target=True)        
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("loading gpt2...")

        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.V = self.gpt2_tokenizer.vocab_size 

        print("finish loading gpt2....")
   
        for param in self.gpt2_model.parameters():
            param.requires_grad = False

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
        x = input['bpe_id']
        b, n = x.shape[:2]
    
        def roots():
            roots = (self.root_emb  @ self.rule_state_emb.t())
            roots = roots.log_softmax(-1)
            return roots.expand(b, roots.shape[-1])

        def terms():            
            term_emb = self.rule_state_emb
            term_prob = ((self.term_mlp(term_emb) + term_emb) @ self.vocab_emb).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None,None], x[:, :, None]], term_prob
        
            # term_prob = term_prob.unsqueeze(0).unsqueeze(1).expand(
            #     b, n, self.T, self.V
            # )
            # indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            # term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            # return term_prob

        def rules():
            split = self.split(self.rule_state_emb).log_softmax(-1)

            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb
            parent1 = self.parent_mlp1(nonterm_emb) + nonterm_emb   

            # parent2 = self.parent_mlp2(nonterm_emb) + nonterm_emb   
            left = (self.left_mlp(rule_state_emb) + rule_state_emb) @  parent1.t()             

            right = (self.right_mlp(rule_state_emb) + rule_state_emb) @ parent1.t()

            # right = left
            # head = head.softmax(-01)

            left = left.softmax(-2)
            P = torch.linalg.inv(torch.diag(left.new_ones(self.NT)) - left* split[:,1].exp().unsqueeze(0))
            
            right = right.softmax(-2) 
            # left_m =  left[:self.NT, :]
            # left_p =  left[self.NT:, :]
            # right_m = right[:self.NT, :]
            # right_p = right[self.NT:, :]            
            return (left, P, right, split)

        root, (unary, all_emission_prob), (left_m,  P, right_m, split) = roots(), terms(), rules()

        return {'unary': unary,
                'root': root,
                # 'head': head,
                'left_m': left_m,
                'right_m': right_m,
                'P': P,
                # 'left_p': left_p,
                # 'right_p' : right_p,
                'split': split,
                'all_emission_prob': all_emission_prob,
                'kl': 0}

    def check_correctness(self, input):
        # marginal = self.pcfg.marginal_next_preterminal(rules=rules, lens=input['seq_len'])['marginal']
        # marginal2 = self.pcfg.conclusion_1999_next_preterminal(rules=rules, lens=input['seq_len'])['marginal']
        # pdb.set_trace()
        pass

    def distill_kl_loss(self, input):
        rules = self.forward(input)        

        bs = rules['unary'].shape[0]
        
        orig_bpe = input["bpe_id"]

        with torch.no_grad():
            bos = torch.LongTensor(bs, 1).fill_(self.gpt2_tokenizer.bos_token_id).to(self.device)
            # eos = torch.LongTensor(bs, 1).fill_(self.gpt2_tokenizer.eos_token_id).to(self.device)            

            bpe = torch.cat([
                bos, orig_bpe
            ], dim=-1)
            
            y = self.gpt2_model(bpe).logits[:, :-1, :]
            y = y.softmax(-1)

        tmp = self.pcfg.marginal_next_preterminal(rules=rules, lens=input['seq_len'])
        pcfg_next_preterminal_prob  = tmp['marginal']

        # logZ = tmp['partition']
        # result =  self.pcfg._inside_triton(rules=rules, lens=input['seq_len'])
        # pdb.set_trace()
        all_emission_prob = rules['all_emission_prob']

        x = ((pcfg_next_preterminal_prob @ all_emission_prob.exp())+1e-9).log()
        
        # x = ((pcfg_next_preterminal_prob+1e-9).log()[...,None] + all_emission_prob[None, None,...] ).logsumexp(-2)

        kl_loss = y * x
        
        gold_loss = -x.gather(-1,orig_bpe.unsqueeze(-1)).sum([-2,-1]).mean()
    
        # pdb.set_trace()        
        # pdb.set_trace()
        return -kl_loss.sum([-1,-2]).mean() + gold_loss

        # kl_loss = torch.kl_div()        
        
        

    def loss(self, input):
        return  self.distill_kl_loss(input)
    


        rules = self.forward(input)
        # result =  self.pcfg._inside_triton(rules=rules, lens=input['seq_len'])
        result = self.pcfg._inside(rules=rules, lens=input['seq_len'])
        # pdb.set_trace()
        logZ =  -result['partition'].mean()
        return logZ 
    

    # def evaluate(self, input, decode_type, **kwargs):
    #     rules = self.forward(input)
    #     if decode_type == 'viterbi':
    #         assert NotImplementedError

    #     elif decode_type == 'mbr':
    #         return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
    #     else:
    #         raise NotImplementedError

    def evaluate(self, input, decode_type, **kwargs):
        """
        Fields of input:
            bpe_id: tensor of the shape (batch, seq_len) with CLS and SEP
            seq_len: tensor of the shape (batch), with CLS and SEP
            chunk: BI encoding of words, without CLS and SEP
            word_len: tensor of the shape (batch), without CLS and SEP
        """
        
        x = input["bpe_id"]
        seq_len_v = input["seq_len"]
        rules = self.forward(input, evaluating=True)

        # extract words in the form of tuple (start, end)
        def extract_words(bpe_wo_cls_sep, word_lens, chunks):
            bs, bpe_len = bpe_wo_cls_sep.size()
            words_list = []
            for b_idx in range(bs):
                chunk = chunks[b_idx]
                # bpe idx to word idx
                words = []
                state = -1
                for bpe_idx in range(bpe_len):
                    bpe_label = chunk[bpe_idx]
                    if bpe_label == 0:
                        continue
                    else:
                        assert bpe_label == 1
                        if state == -1:
                            state = bpe_idx  # new begin
                        else:
                            words.append((state, bpe_idx))
                            state = bpe_idx  # new begin
                words.append((state, bpe_len))  # last words
                assert len(words) == word_lens[b_idx]
                words_list.append(words)
            return words_list

        words_list = extract_words(x, input["word_len"], input["chunk"]) 

        # # run the infernce over bpe tokens
        # if decode_type == "viterbi":
        #     # viterbi does not support chunks yet
        #     result = self.pcfg.decode(rules=rules, lens=seq_len_v, viterbi=True, mbr=False)
        # elif decode_type == "mbr":
        result = self.pcfg.decode(rules=rules, lens=seq_len_v, viterbi=False, mbr=True, chunk=words_list)

        # else:
            # raise NotImplementedError        
        # convert bpe-level trees to word-level trees
        ## current implementations assumes that bpe tokens from the same word froms a tree
        predictions_list = result["prediction"]
        new_predictions_list = []
        for b_idx in range(x.size(0)):
            predictions = predictions_list[b_idx]
            words = words_list[b_idx]

            bpe_idx2word_idx = {}
            for word_idx, (start, end) in enumerate(words):
                for _idx in range(start, end):
                    bpe_idx2word_idx[_idx] = word_idx

            new_predictions = []
            for (bpe_start, bpe_end_exclusive) in predictions:
                bpe_end = bpe_end_exclusive - 1
                word_start, word_end = bpe_idx2word_idx[bpe_start], bpe_idx2word_idx[bpe_end]

                word_end_exclusive = word_end + 1
                if (word_start, word_end_exclusive) not in new_predictions:
                    new_predictions.append((word_start, word_end_exclusive))

            new_predictions_list.append(new_predictions)

        result["prediction"] = new_predictions_list
        return result
