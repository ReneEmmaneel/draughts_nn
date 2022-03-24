from training import train
from types import SimpleNamespace
import argparse
import os
import json
from training import train

def load_config_file(args):
    config_file = f'{args.run_folder_str}/config_file.json'
    f = open(config_file)
    data = json.load(f)
    for d in data:
        if d == 'num_training_loops': #this value will always be overriden
            continue
        vars(args)[d] = data[d]
    f.close()
    return args

def continue_from(args):
    args = load_config_file(args)
    train(args, args.run_folder_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder_str', default="", type=str, required=True)
    parser.add_argument('--num_training_loops', default=50, type=int,
                        help='amount of self play and optimising loops')
    args = parser.parse_args()
    continue_from(args)
