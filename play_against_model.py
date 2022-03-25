from training import train
from types import SimpleNamespace
import argparse
import os
import json
import torch
import training
from continue_training import load_config_file
from draughts import play_game_against_model

def load_config_file(args):
    config_file = f'{args.run_folder_str}/config_file.json'
    f = open(config_file)
    data = json.load(f)
    for d in data:
        vars(args)[d] = data[d]
    f.close()
    return args

def play_against_model(args):
    args = load_config_file(args)

    iteration_num = args.iteration if args.iteration >= 0 else args.num_training_loops - 1

    model_filename = f'{args.run_folder_str}/models/iteration_{iteration_num}'
    model = training.PositionEvaluator()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_filename))

    play_game_against_model(model, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_folder_str', default="", type=str, required=True)
    parser.add_argument('--iteration', default=-1, type=int,
                        help='which iteration number to play against. -1 means last model')
    args = parser.parse_args()
    play_against_model(args)
