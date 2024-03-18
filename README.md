# TN-PCFG and NBL-PCFG

source code of  
- NAACL2021 [PCFGs Can Do Better: Inducing Probabilistic Context-Free Grammars with Many Symbols](https://www.aclweb.org/anthology/2021.naacl-main.117.pdf) 
- ACL2021 [Neural Bi-Lexicalzied PCFG Induction](http://faculty.sist.shanghaitech.edu.cn/faculty/tukw/acl21pcfg.pdf).
- NAACL2022 [Dynamic Programming in Rank Space: Scaling Structured Inference with Low-Rank HMMs and PCFGs 
](https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/naacl22rank.pdf).
- EMNLP2023 Findings [Simple Hardware-Efficient PCFGs with Independent Left and Right Productions](https://aclanthology.org/2023.findings-emnlp.113.pdf)

The repository also contain faster implementations of:

-  [Compound PCFG](https://www.aclweb.org/anthology/P19-1228/)
-  [Neural Lexicalized PCFG](https://www.aclweb.org/anthology/2020.tacl-1.42/)


## News
- 23/12: We upload the [simple PCFG](https://aclanthology.org/2023.findings-emnlp.113.pdf)) code.

- 22/04: Our paper [Dynamic Programming in Rank Space: Scaling Structured Inference with Low-Rank HMMs and PCFGs](https://openreview.net/forum?id=KBpfIEHa9Th) has been accepted to NAACL2022.

- 22/04: We highly optimize the implementation of the inside algorithms. We leverage the [log-einsum-exp trick](https://arxiv.org/abs/2004.06231) to avoid expensive logsumexp operations.
           
## Setup

setup environment 

```
conda create -n pcfg python=3.9
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
If you find this repository helpful, please cite our work.


```
@inproceedings{liu-etal-2023-simple,
    title = "Simple Hardware-Efficient {PCFG}s with Independent Left and Right Productions",
    author = "Liu, Wei  and
      Yang, Songlin  and
      Kim, Yoon  and
      Tu, Kewei",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.113",
    doi = "10.18653/v1/2023.findings-emnlp.113",
    pages = "1662--1669",
    abstract = "Scaling dense PCFGs to thousands of nonterminals via low-rank parameterizations of the rule probability tensor has been shown to be beneficial for unsupervised parsing. However, PCFGs scaled this way still perform poorly as a language model, and even underperform similarly-sized HMMs. This work introduces $\emph{SimplePCFG}$, a simple PCFG formalism with independent left and right productions. Despite imposing a stronger independence assumption than the low-rank approach, we find that this formalism scales more effectively both as a language model and as an unsupervised parser. We further introduce $\emph{FlashInside}$, a hardware IO-aware implementation of the inside algorithm for efficiently scaling simple PCFGs. Through extensive experiments on multiple grammar induction benchmarks, we validate the effectiveness of simple PCFGs over low-rank baselines.",
}

@inproceedings{yang-etal-2022-dynamic,
    title = "Dynamic Programming in Rank Space: Scaling Structured Inference with Low-Rank {HMM}s and {PCFG}s",
    author = "Yang, Songlin  and
      Liu, Wei  and
      Tu, Kewei",
    editor = "Carpuat, Marine  and
      de Marneffe, Marie-Catherine  and
      Meza Ruiz, Ivan Vladimir",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.353",
    doi = "10.18653/v1/2022.naacl-main.353",
    pages = "4797--4809",
    abstract = "Hidden Markov Models (HMMs) and Probabilistic Context-Free Grammars (PCFGs) are widely used structured models, both of which can be represented as factor graph grammars (FGGs), a powerful formalism capable of describing a wide range of models. Recent research found it beneficial to use large state spaces for HMMs and PCFGs. However, inference with large state spaces is computationally demanding, especially for PCFGs. To tackle this challenge, we leverage tensor rank decomposition (aka. CPD) to decrease inference computational complexities for a subset of FGGs subsuming HMMs and PCFGs. We apply CPD on the factors of an FGG and then construct a new FGG defined in the rank space. Inference with the new FGG produces the same result but has a lower time complexity when the rank size is smaller than the state size. We conduct experiments on HMM language modeling and unsupervised PCFG parsing, showing better performance than previous work. Our code is publicly available at \url{https://github.com/VPeterV/RankSpace-Models}.",
}

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









