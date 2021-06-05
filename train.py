# -*- coding: utf-8 -*-

import argparse
import os
from parser.cmds import Evaluate, Train
import shutil
import torch
import traceback
from pathlib import Path
from easydict import EasyDict as edict
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PCFGs'
    )
    parser.add_argument('--conf', '-c', default='')
    parser.add_argument('--device', '-d', default='0')


    args2 = parser.parse_args()
    yaml_cfg = yaml.load(open(args2.conf, 'r'))
    args = edict(yaml_cfg)
    args.update(args2.__dict__)

    print(f"Set the device with ID {args.device} visible")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    config_path = Path(args.conf  if args.conf else args2.load_from_dir + "/config.yaml")
    config_name = config_path.stem
    args.save_dir = args.save_dir + "/{}".format(config_name)

    try:
        command = Train()
        command(args)
    except KeyboardInterrupt:
        command = int(input('Enter 0 to delete the repo, and enter anything else to save.'))
        if command == 0:
            shutil.rmtree(args.save_dir)
            print("You have successfully delete the created log directory.")
        else:
            print("log directory have been saved.")
    except Exception:
        traceback.print_exc()
        shutil.rmtree(args.save_dir)
        print("log directory have been deleted.")

