import re

import draughts
import random
import torch
import utils
from torch.utils.data import Dataset, DataLoader

def MCTS(model, board_state, is_white, run_time=1600):
    model.eval()
    is_white_tensor = torch.unsqueeze(torch.Tensor([int(is_white)*2-1]), dim=0)
    value, move_probabilities = model(torch.unsqueeze(board_state, dim=0), is_white_tensor)

    moves_dict = dict(draughts.possible_moves(board_state, is_white))

    moves_prob = []

    #TODO: MCTS, currently just taking most probably move
    for move in list(moves_dict.keys()):
        move_positions = re.split('[x\-]',move)
        fy, fx = utils.pos_to_coor(move_positions[0])
        ty, tx = utils.pos_to_coor(move_positions[-1])
        probability = move_probabilities[0][fy][fx][ty][tx]
        moves_prob.append((move, probability.item()))

    moves_prob.sort(key=lambda x:x[1], reverse=True)

    return moves_prob[0][0], move_probabilities.squeeze()

def play_games(model, k=1000, max_length=100):
    """Play k games of given max_length.
    return list of (gamestate, current_player, outcome) tuples
    """
    all_states = []
    for i in range(k):
        states_current_game = []
        position = draughts.load_position()
        game_over = False
        win_value = 0

        while win_value == 0 and len(states_current_game) < max_length:
            moves_dict = dict(draughts.possible_moves(*position))
            if len(moves_dict.keys()) == 0:
                break

            move, search_probabilities = MCTS(model, *position)

            position = (moves_dict[move], not position[1])
            states_current_game.append((*position, search_probabilities))
            win_value = draughts.check_win(position[0])

        if win_value == 0 and len(states_current_game) < max_length:
            win_value = int(position[1]) * -2 + 1

        all_states = all_states + [(_[0], torch.Tensor([int(_[1])*2-1]), win_value, search_probabilities) for _ in states_current_game]
    return all_states

class GameStateDataset(Dataset):
    """GameStateDataset"""

    def __init__(self, data, transform=None):
        """
        Args:
            data (list): List with gamestate data as tuples.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    #Example of self play and creating a dataset
    game_states = play_games(k=3)
    game_state_dataset = GameStateDataset(game_states)
    dataloader = DataLoader(game_state_dataset, batch_size=4, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
