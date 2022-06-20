# TN-PCFG and NBL-PCFG

source code of  
- NAACL2021 [PCFGs Can Do Better: Inducing Probabilistic Context-Free Grammars with Many Symbols](https://www.aclweb.org/anthology/2021.naacl-main.117.pdf) 
- ACL2021 [Neural Bi-Lexicalzied PCFG Induction](http://faculty.sist.shanghaitech.edu.cn/faculty/tukw/acl21pcfg.pdf).
- NAACL2022 [Dynamic Programming in Rank Space: Scaling Structured Inference with Low-Rank HMMs and PCFGs (https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/naacl22rank.pdf).

The repository also contain faster implementations of:

-  [Compound PCFG](https://www.aclweb.org/anthology/P19-1228/)
-  [Neural Lexicalized PCFG](https://www.aclweb.org/anthology/2020.tacl-1.42/)


## News
- 22/04: Our paper [Dynamic Programming in Rank Space: Scaling Structured Inference with Low-Rank HMMs and PCFGs](https://openreview.net/forum?id=KBpfIEHa9Th) has been accepted to NAACL2022.

- 22/04: We highly optimize the implementation of the inside algorithms. We leverage the [log-einsum-exp trick](https://arxiv.org/abs/2004.06231) to avoid expensive logsumexp operations.
           
## Setup

prepare environment 

```
conda create -n pcfg python=3.7
conda activate pcfg
while read requirement; do pip install $requirement; done < requirement.txt 
```

prepare dataset

You can download the dataset and pretrained model (TN-PCFG and NBL-PCFG) from:  https://mega.nz/folder/OU5yiTjC#oeMYj1gBhqm2lRAdAvbOvw

PTB:  ptb_cleaned.zip / CTB and SPRML: ctb_sprml_clean.zip

You can directly use the propocessed pickle file or create pickle file by your own

```
python  preprocessing.py  --train_file path/to/your/file --val_file path/to/your/file --test_file path/to/your/file  --cache path/
```

After this, your data folder should look like this:

```
config/
   ├── tnpcfg_r500_nt250_t500_curriculum0.yaml
   ├── ...
  
data/
   ├── ptb-train-lpcfg.pickle    
   ├── ptb-val-lpcfg.pickle
   ├── ptb-test-lpcfg.pickle
   ├── ...
   
log/
fastNLP/
parser/
train.py
evaluate.py
preprocessing.py
```



## Train

**TN-PCFG**

python train.py  --conf tnpcfg_r500_nt250_t500_curriculum0.yaml

**Compound PCFG**

python train.py --conf cpcfg_nt30_t60_curriculum1.yaml

....

## Evaluation

For example, the saved directory should look like this:

```
log/
   ├── NBLPCFG2021-01-26-07_47_29/
   	  ├── config.yaml
   	  ├── best.pt
   	  ├── ...
```

python evaluate.py --load_from_dir log/NBLPCFG2021-01-26-07_47_29  --decode_type mbr --eval_dep 1 


## Contact

If you have any question, plz contact bestsonta@gmail.com. 

## Citation

If these codes help you, plz cite our paper:

```
@inproceedings{yang-etal-2021-neural,
    title = "Neural Bi-Lexicalized {PCFG} Induction",
    author = "Yang, Songlin  and
      Zhao, Yanpeng  and
      Tu, Kewei",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.209",
    doi = "10.18653/v1/2021.acl-long.209",
    pages = "2688--2699",
    abstract = "Neural lexicalized PCFGs (L-PCFGs) have been shown effective in grammar induction. However, to reduce computational complexity, they make a strong independence assumption on the generation of the child word and thus bilexical dependencies are ignored. In this paper, we propose an approach to parameterize L-PCFGs without making implausible independence assumptions. Our approach directly models bilexical dependencies and meanwhile reduces both learning and representation complexities of L-PCFGs. Experimental results on the English WSJ dataset confirm the effectiveness of our approach in improving both running speed and unsupervised parsing performance.",
}

@inproceedings{yang-etal-2021-pcfgs,
    title = "{PCFG}s Can Do Better: Inducing Probabilistic Context-Free Grammars with Many Symbols",
    author = "Yang, Songlin  and
      Zhao, Yanpeng  and
      Tu, Kewei",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.117",
    pages = "1487--1498",
    abstract = "Probabilistic context-free grammars (PCFGs) with neural parameterization have been shown to be effective in unsupervised phrase-structure grammar induction. However, due to the cubic computational complexity of PCFG representation and parsing, previous approaches cannot scale up to a relatively large number of (nonterminal and preterminal) symbols. In this work, we present a new parameterization form of PCFGs based on tensor decomposition, which has at most quadratic computational complexity in the symbol number and therefore allows us to use a much larger number of symbols. We further use neural parameterization for the new form to improve unsupervised parsing performance. We evaluate our model across ten languages and empirically demonstrate the effectiveness of using more symbols.",
}
```
## Ack.
We use [fastNLP](https://github.com/fastnlp/fastNLP) and the code template of [Supar](https://github.com/yzhangcs/parser)









