import time
import os
import logging
from distutils.dir_util import copy_tree

from parser.model import NeuralPCFG, CompoundPCFG, TNPCFG, NeuralBLPCFG, NeuralLPCFG, FastTNPCFG, FastNBLPCFG

from parser.model.simple_C_PCFG import Simple_C_PCFG
from parser.model.simple_N_PCFG import Simple_N_PCFG
from parser.model.simple_N_PCFG_split import Simple_N_PCFG_split
from parser.model.autoregressive_PCFG import Autoregressive_PCFG


import torch


def get_model(args, dataset):
    if args.model_name == 'NPCFG':
        return NeuralPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'CPCFG':
        return CompoundPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'simple_CPCFG':
        return Simple_C_PCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'simple_NPCFG':
        return Simple_N_PCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'simple_NPCFG_split':
        return Simple_N_PCFG_split(args, dataset).to(dataset.device)

    elif args.model_name == 'TNPCFG':
        return TNPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'arpcfg':
        return Autoregressive_PCFG(args, dataset).to(dataset.device)


    elif args.model_name == 'NLPCFG':
        return NeuralLPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'NBLPCFG':
        return NeuralBLPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'FastTNPCFG':
        return FastTNPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'FastNBLPCFG':
        return FastNBLPCFG(args, dataset).to(dataset.device)

    else:
        raise KeyError


def get_optimizer(args, model):
    if args.name == 'adam':
        return torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu))
    elif args.name == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

def get_logger(args, log_name='train',path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    handler = logging.FileHandler(os.path.join(args.save_dir if path is None else path, '{}.log'.format(log_name)), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info(args)
    return logger


def create_save_path(args):
    model_name = args.model.model_name
    suffix = "/{}".format(model_name) + time.strftime("%Y-%m-%d-%H_%M_%S",
                                                                             time.localtime(time.time()))
    from pathlib import Path
    saved_name = Path(args.save_dir).stem + suffix
    args.save_dir = args.save_dir + suffix

    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print('Creating {}'.format(args.save_dir))
        os.makedirs(args.save_dir)
    # save the config file and model file.
    import shutil
    shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
    os.makedirs(args.save_dir + "/parser")
    copy_tree("parser/", args.save_dir + "/parser")
    return  saved_name

