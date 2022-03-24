import argparse
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json
from tqdm import *
import multiprocessing as mp

from self_play import play_games, GameStateDataset, model_vs

class PositionEvaluator(nn.Module):
    def __init__(self):
        super().__init__()

        self.c_hidden = 10

        self.input = nn.Sequential(
            nn.Conv2d(6, self.c_hidden, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.c_hidden),
            nn.Tanh(),
            nn.Conv2d(self.c_hidden, self.c_hidden*2, stride=(2,1), kernel_size=3, bias=True),
            nn.BatchNorm2d(self.c_hidden*2),
            nn.Tanh(),
            nn.Conv2d(self.c_hidden*2, self.c_hidden*5, stride=1, kernel_size=(4,3), bias=True),
            nn.BatchNorm2d(self.c_hidden*5),
            nn.Tanh()
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.c_hidden*5,1),
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.c_hidden*5,self.c_hidden*5),
            nn.ReLU(),
            nn.Linear(self.c_hidden*5,50*50)
        )

    def forward(self, board_state, is_white):
        """Given a [B,H,W,P] board_state and [B] is_white
        return [B] value score and [B,H,W,H,W] policy head
        """
        W = 5
        H = 10

        is_white = is_white.squeeze().repeat(H*W).reshape(-1,H,W,1)
        board_state = torch.cat((board_state, is_white), dim=3)

        board_state = board_state.permute(0,3,1,2)

        output = self.input(board_state)
        if torch.isnan(output).any():
            raise NotImplementedError()

        policy = self.policy_head(output).reshape(-1,H,W,H,W)
        value = self.value_head(output).reshape(-1)

        return value, policy

    @property
    def device(self):
        return next(self.parameters()).device

def get_dataloader(run_folder, previous_states = []):
    saved_positions_folder = f'{run_folder}/saved_positions'
    #Load previous states
    if os.path.exists(saved_positions_folder):
        all_files = []
        for filename in os.listdir(saved_positions_folder):
            all_files.append(filename.split('.')[0])
        #Only look at previous n states
        for file in sorted(all_files[0:args.keep_previous_n_games]):
            with open(f'{saved_positions_folder}/{file}.pkl', 'rb') as inp:
                previous_states = previous_states + pickle.load(inp)
    else:
        os.makedirs(saved_positions_folder)

    #Make dataset
    game_state_dataset = GameStateDataset(previous_states)
    dataloader = DataLoader(game_state_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    return dataloader

def self_play_and_dataset(run_folder, model, pool, args):
    """Self play and create new dataset
    input:
        run_folder - str with path to folder to save previous states
        model - latest model
        args.keep_previous_n_games - how many previous saved states to add to newly generated states
    """
    all_new_states = play_games(model, pool, args)
    previous_states = all_new_states
    saved_positions_folder = f'{run_folder}/saved_positions'

    dataloader = get_dataloader(run_folder, previous_states=previous_states)

    #Save current generated states
    amount = len(os.listdir(saved_positions_folder))
    with open(f'{saved_positions_folder}/{amount:09d}.pkl', 'wb') as outp:
        pickle.dump(all_new_states, outp, pickle.HIGHEST_PROTOCOL)

    return dataloader

@torch.no_grad()
def test_model(model, data_loader, beta=1):
    """
    Function for testing a model on a dataset.
    Inputs:
        model - model to test
        data_loader - Data Loader for the dataset you want to test on.
    Outputs:
        average_loss - Average loss
    """
    total_loss, total_policy_loss, total_value_loss = 0., 0., 0.
    num_samples = 0

    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.BCELoss()
    for batch_ndx, sample in enumerate(data_loader):
        model.eval()

        board_state          = sample[0].to(model.device)
        is_white             = sample[1].to(model.device)
        outcome              = sample[2].to(model.device)
        search_probabilities = sample[3].to(model.device)

        batch_size = len(sample[0])

        with torch.no_grad():
            value, policy = model(board_state, is_white)

        value_loss = value_loss_fn(value, outcome.float())

        policy = torch.softmax(policy.view(batch_size, -1), axis=1)
        search_probabilities = search_probabilities.view(batch_size, -1)

        policy_loss = policy_loss_fn(policy, search_probabilities.detach())
        loss = value_loss + beta * policy_loss

        total_loss += loss * batch_size
        total_policy_loss += policy_loss * batch_size
        total_value_loss += value_loss * batch_size

        num_samples += batch_size

    average_loss = total_loss / num_samples
    average_value_loss = total_policy_loss / num_samples
    average_policy_loss = total_value_loss / num_samples
    return average_loss, average_value_loss, average_policy_loss

def train_evaluator(model, data_loader, optimizer, args, beta=1):
    """
    Function for training a model on a dataset. Train the model for one epoch.
    Inputs:
        model - model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizer - The optimizer used to update the parameters
    Outputs:
        average_loss - Average loss
    """

    total_loss = 0.
    num_samples = 0

    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.BCELoss()

    for epochs in trange(args.epochs_per_train_evaluator, leave=False, disable=not args.verbose):
        for batch_ndx, sample in tqdm(enumerate(data_loader), desc='Train evaluator model', leave=False) if args.verbose else enumerate(data_loader):
            model.train()
            board_state          = sample[0].to(model.device)
            is_white             = sample[1].to(model.device)
            outcome              = sample[2].to(model.device)
            search_probabilities = sample[3].to(model.device)

            batch_size = len(sample[0])

            optimizer.zero_grad()
            value, policy = model(board_state, is_white)
            value_loss = value_loss_fn(value, outcome.float())

            policy = torch.softmax(policy.view(batch_size, -1), axis=1)
            search_probabilities = search_probabilities.view(batch_size, -1)

            policy_loss = policy_loss_fn(policy, search_probabilities.detach())
            loss = value_loss + beta * policy_loss

            loss.backward()
            optimizer.step()

            total_loss += loss * batch_size
            num_samples += batch_size

    average_loss = total_loss / num_samples
    return average_loss

def save_model(model, folder, i):
    """
    Input:
        model: model to save
        folder: folder to save to
        i: iteration number
    """
    foldername = f'{folder}/models'
    filename = f'{foldername}/iteration_{i}'

    #Load previous states
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    torch.save(model.state_dict(), filename)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_config_file(run_folder_str, args):
    config_file = f'{run_folder_str}/config_file.json'

    with open(config_file, "w") as outfile:
        json.dump(vars(args), outfile, indent=2)

def before_training(args):
    """Function to be called, only when starting training for the first time,
    meaning it is not called when started from continue_training.py
    """
    if not os.path.exists(args.training_folder):
        os.makedirs(args.training_folder)

    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_folder_str = f'{args.training_folder}/{time_string}'
    os.makedirs(run_folder_str)

    write_config_file(run_folder_str, args)
    return run_folder_str

def train(args, run_folder_str):
    """Function that creates the model, and finishes the training loop.
    This loop includes selfplay and training the neural network.
    If the run_folder already contains previous models, use the latest model.
    """
    model = PositionEvaluator()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    #Load latest model
    saved_models_folder = f'{run_folder_str}/models'
    all_models = []
    if os.path.exists(saved_models_folder):
        for filename in os.listdir(saved_models_folder):
            all_models.append(int(filename.split('_')[1]))
        all_models.sort(reverse=True)
        #load latest model, if applicable
        if len(all_models) > 0:
            model.load_state_dict(torch.load(f'{saved_models_folder}/iteration_{all_models[0]}'))

    #Only allow parallel execution if we are not continuing to train,
    #somehow, multiprocessing does not work when continue training :(
    #TODO: fix it so that multiprocessing does work when continue training
    parallel=len(all_models) == 0
    if parallel:
        pool = mp.Pool()

    iterator = range(len(all_models), args.num_training_loops)
    for i in (tqdm(iterator, desc='Training') if args.verbose else iterator):
        current_dataloader = self_play_and_dataset(run_folder_str, model, pool if parallel else None, args)
        train_evaluator(model, current_dataloader, optimizer, args)
        save_model(model, run_folder_str, i)
        loss, value_loss, policy_loss = test_model(model, current_dataloader)
        with open(f'{run_folder_str}/losses.txt', "a") as file:
            file.write(f'{i}\t{loss}\t{value_loss}\t{policy_loss}\n')

    if parallel:
        pool.close()
        pool.join()

def evaluate(args, run_folder_str):
    last_model_filename = f'{run_folder_str}/models/iteration_{args.num_training_loops - 1}'
    last_model = PositionEvaluator()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    last_model = last_model.to(device)
    last_model.load_state_dict(torch.load(last_model_filename))

    current_dataloader = get_dataloader(run_folder_str)

    for i in range(args.num_training_loops):
        filename = f'{run_folder_str}/models/iteration_{i}'
        curr_model = PositionEvaluator()
        curr_model.load_state_dict(torch.load(filename))
        loss, value_loss, policy_loss = test_model(curr_model, current_dataloader)
        with open(f'{run_folder_str}/final_losses.txt', "a") as file:
            file.write(f'{i}\t{loss}\t{value_loss}\t{policy_loss}\n')
        with open(f'{run_folder_str}/final_tournament.txt', "a") as file:
            file.write(f'last vs {i}:\t{model_vs(last_model, curr_model, args, num_games=20)}\n')

def main(args):
    run_folder_str = before_training(args)
    train(args, run_folder_str)
    evaluate(args, run_folder_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Directories
    parser.add_argument('--training_folder', default='training', type=str,
                        help='folder to keep training specific data')

    #Hyperparams
    parser.add_argument('--num_training_loops', default=50, type=int,
                        help='amount of self play and optimising loops')
    parser.add_argument('--MCTS_run_time', default=1600, type=int,
                        help='how many leave nodes to traverse to')
    parser.add_argument('--temperature', default=1., type=float,
                        help='value controlling exploration in MCTS')
    parser.add_argument('--generate_k_games', default=100, type=int,
                        help='number of games to generated during self_play')
    parser.add_argument('--max_length_games', default=200, type=int,
                        help='max length of a game during self_play')
    parser.add_argument('--keep_previous_n_games', default=5, type=int,
                        help='how many retraining cycles to keep previous games for')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size for training the neural network')
    parser.add_argument('--epochs_per_train_evaluator', default=4, type=int,
                        help='epochs per evaluator training cycle')

    #Misc
    parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args)
