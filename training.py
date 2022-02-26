import argparse
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from self_play import play_games, GameStateDataset

class PositionEvaluator(nn.Module):
    def __init__(self):
        super().__init__()

        self.c_hidden = 10

        self.input = nn.Sequential(
            nn.Conv2d(5, self.c_hidden, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.c_hidden),
            nn.Tanh(),
            nn.Conv2d(self.c_hidden, self.c_hidden, stride=(1,2), kernel_size=3, bias=True),
            nn.BatchNorm2d(self.c_hidden),
            nn.Tanh(),
            nn.Conv2d(self.c_hidden, self.c_hidden, stride=1, kernel_size=(3,4), bias=True),
            nn.BatchNorm2d(self.c_hidden),
            nn.Tanh()
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(self.c_hidden + 1,1),
            nn.Tanh()
        )

    def forward(self, board_state, is_white):
        board_state = board_state.permute(0,2,3,1)
        board_state = self.input(board_state)
        board_state = board_state.reshape(-1, self.c_hidden)
        board_state = torch.cat((board_state, is_white), dim=1)

        value = self.fully_connected(board_state).reshape(-1)

        return value

    @property
    def device(self):
        return next(self.parameters()).device

def self_play_and_dataset(run_folder, n=5):
    """Self play and create new dataset
    input:
        run_folder - str with path to folder to save previous states
        n - how many previous saved states to add to newly generated states
    """
    all_new_states = play_games(k=3)
    previous_states = all_new_states
    saved_positions_folder = f'{run_folder}/saved_positions'

    #Load previous states
    if os.path.exists(saved_positions_folder):
        all_files = []
        for filename in os.listdir(saved_positions_folder):
            all_files.append(filename.split('.')[0])
        #Only look at previous n states
        for file in sorted(all_files[0:n]):
            with open(f'{saved_positions_folder}/{file}.pkl', 'rb') as inp:
                previous_states = previous_states + pickle.load(inp)
    else:
        os.makedirs(saved_positions_folder)

    #Save current generated states
    amount = len(os.listdir(saved_positions_folder))
    with open(f'{saved_positions_folder}/{amount:09d}.pkl', 'wb') as outp:
        pickle.dump(all_new_states, outp, pickle.HIGHEST_PROTOCOL)

    #Make dataset
    game_state_dataset = GameStateDataset(previous_states)
    dataloader = DataLoader(game_state_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    return dataloader

@torch.no_grad()
def test_model(model, data_loader):
    """
    Function for testing a model on a dataset.
    Inputs:
        model - model to test
        data_loader - Data Loader for the dataset you want to test on.
    Outputs:
        average_loss - Average loss
    """
    total_loss = 0.
    num_samples = 0
    loss_fn = nn.MSELoss()
    for batch_ndx, sample in enumerate(data_loader):
        model.eval()
        board_state, is_white, outcome = sample[0].to(model.device), sample[1].to(model.device), sample[2].to(model.device)
        with torch.no_grad():
            prediction = model(board_state, is_white)
        loss = loss_fn(prediction, outcome)

        batch_size = len(sample[0])

        total_loss += loss * batch_size

        num_samples += batch_size

    average_loss = total_loss / num_samples
    return average_loss

def train_evaluator(model, data_loader, optimizer):
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

    loss_fn = nn.MSELoss()
    for batch_ndx, sample in enumerate(data_loader):
        model.train()
        board_state, is_white, outcome = sample[0].to(model.device), sample[1].to(model.device), sample[2].to(model.device).float()

        optimizer.zero_grad()
        prediction = model(board_state, is_white)
        loss = loss_fn(prediction, outcome)

        loss.backward()
        optimizer.step()

        batch_size = len(sample[0])

        total_loss += loss * batch_size
        num_samples += batch_size

    average_loss = total_loss / num_samples
    return average_loss

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    if not os.path.exists(args.training_folder):
        os.makedirs(args.training_folder)

    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_folder_str = f'{args.training_folder}/{time_string}'
    os.makedirs(run_folder_str)

    model = PositionEvaluator()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for i in range(50):
        current_dataloader = self_play_and_dataset(run_folder_str)
        train_evaluator(model, current_dataloader, optimizer)
        loss = test_model(model, current_dataloader)
        print(f'{i} {loss}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_folder', default='training', type=str,
                        help='folder to keep training specific data')

    args = parser.parse_args()
    train(args)
