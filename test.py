from parser.pcfgs.pcfg import PCFG, Faster_PCFG
import torch
import time

if __name__ == '__main__':
    b = 10
    n = 30
    nt = 30
    t = 60
    pcfg1 = PCFG()
    pcfg2 = Faster_PCFG()
    rule = {}
    unary = torch.rand(b, n, t)
    binary = torch.rand(b, nt, nt+t, nt+t)
    root = torch.rand(b, nt)
    rule['unary'] = unary
    rule['rule'] = binary
    rule['root'] = root
    len = torch.zeros(b).fill_(n).long()
    for _ in range(10):
        pcfg1._inside(rule, len)
    start = time.time()
    for _ in range(100):
        pcfg1._inside(rule, len)
    print(time.time() - start)
    start = time.time()
    for _ in range(100):
        pcfg2._inside(rule, len)
    print(time.time() - start)


