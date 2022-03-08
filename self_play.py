import re

import draughts
import random
import torch
import utils
from numpy.random import choice
from torch.utils.data import Dataset, DataLoader

def bool_to_tensor(bool):
    return torch.unsqueeze(torch.Tensor([int(bool)*2-1]), dim=0)

class Node():
    def __init__(self, board_state, is_white, P):
        self.board_state = board_state
        self.is_white = is_white
        self.is_white_tensor = bool_to_tensor(is_white)

        self.N = 0 #num times traversed
        self.W = 0. #total value next state
        self.Q = 0. #W / N, avg value next state
        self.P = P #prior of selecting this node
        self.is_end = False #True if leave node for the entire game

        self.children = []

    def add_child(self, notation, child_node):
        self.children.append((notation, child_node))

def MCTS(model, subtree, board_state, is_white, args, deterministic=False):
    """Monte Carlo Tree Search (MCTS)
    Pretty cool algorithm, instead of alpha-beta pruning or some other tree search
    that only looks are the value of each nodes, it uses the policy head to estimate
    which nodes are good to look at.

    For each node, store N (how many times has this action been taken),
    W (total value of next state), Q (mean value of next state), P (prior of selecting this node)

    Input:
        model: neural network, takes in the board state, returns value and policy
        subtree: if not None, use this subtree to store already calculated positions
        board_state: the current board
        is_white: boolean that's true if it is whites turn
        run_time: how many leaves are looked at
        deterministic: if true, return highest probability move, otherwise randomly sample
        temperature: value that encourages exploration
    Output:
        next_move: notation of the next move
        move_probabilities: real move probabilities (used to train the policy head)
    """
    model.eval()
    is_white_tensor = bool_to_tensor(is_white)
    value, move_probabilities = model(torch.unsqueeze(board_state, dim=0), is_white_tensor)

    moves_dict = dict(draughts.possible_moves(board_state, is_white))

    if subtree is None:
        root = Node(board_state, is_white, 1)
    else:
        root = subtree

    run_time = args.MCTS_run_time - root.N

    def add_children(node, moves_dict):
        #Given a node, loop through all possible moves and add child nodes, including their priors
        tot_prob = 0.
        for move in list(moves_dict.keys()):
            move_positions = re.split('[x\-]',move)
            fy, fx = utils.pos_to_coor(move_positions[0])
            ty, tx = utils.pos_to_coor(move_positions[-1])
            probability = move_probabilities[0][fy][fx][ty][tx]
            tot_prob += probability

            node.add_child(move, Node(moves_dict[move], not node.is_white, probability))
        for _, node in root.children:
            node.P /= tot_prob

    #add leave nodes
    if len(root.children) == 0:
        add_children(root, moves_dict)

    for i in range(run_time):
        traverstion = [root]

        while len(traverstion[-1].children) > 0:
            #1. choose action to max Q+U, traverse to node
            #2. repeat until found leave
            max_child = max(traverstion[-1].children, key=lambda c:c[1].Q + c[1].P / (1 + c[1].N))
            traverstion.append(max_child[1])

        #3. calculate v, p using model, and add children
        max_child = traverstion[-1]
        v, p = model(torch.unsqueeze(max_child.board_state, dim=0), max_child.is_white_tensor)

        moves_dict = dict(draughts.possible_moves(max_child.board_state, max_child.is_white))
        add_children(max_child, moves_dict)

        #3b. if leave of entire game (no possible moves), set node.is_end to True and already add the values
        if len(max_child.children) == 0:
            win_value = draughts.check_win(max_child.board_state)
            max_child.N = 1
            max_child.W = win_value
            max_child.Q = win_value
            max_child.is_end = True

        #4. backup previous edges:
        for node in traverstion[-2::-1]:
            node.N += 1
            node.W += v.item()
            node.Q = node.W / node.N

    moves_with_n = [(child[0], child[1].N) for child in root.children]
    probabilities = [p[1] ** (1/args.temperature) for p in moves_with_n]
    probabilities = [float(i)/sum(probabilities) for i in probabilities]

    if deterministic:
        next_move = max(moves_with_n, key=lambda c:c[0])[0]
    else:
        next_move = choice([p[0] for p in moves_with_n], 1, p=probabilities)[0]

    real_move_probabilities = torch.zeros_like(move_probabilities)

    next_node = None
    for i, child in enumerate(root.children):
        if next_move == child[0]:
            next_node = child[1]
        move_positions = re.split('[x\-]',child[0])
        fy, fx = utils.pos_to_coor(move_positions[0])
        ty, tx = utils.pos_to_coor(move_positions[-1])
        real_move_probabilities[0][fy][fx][ty][tx] = probabilities[i]
    assert next_node

    return next_move, real_move_probabilities.squeeze(), next_node

def play_games(model, args):
    """Play k games of given max_length.
    return list of (gamestate, current_player, outcome) tuples
    """
    k = args.generate_k_games


    all_states = []

    for i in range(k):
        states_current_game = []
        position = draughts.load_position()
        game_over = False
        win_value = 0
        subtree = None

        while win_value == 0 and len(states_current_game) < args.max_length_games:
            moves_dict = dict(draughts.possible_moves(*position))
            if len(moves_dict.keys()) == 0:
                break

            move, search_probabilities, subtree = MCTS(model, subtree, *position, args)

            position = (moves_dict[move], not position[1])
            states_current_game.append((*position, search_probabilities))
            win_value = draughts.check_win(position[0])

        if win_value == 0 and len(states_current_game) < args.max_length_games:
            win_value = int(position[1]) * -2 + 1

        all_states = all_states + [(_[0], torch.Tensor([int(_[1])*2-1]), win_value, search_probabilities) for _ in states_current_game]
    return all_states

def model_vs(model1, model2, max_length=200, num_games=20):
    """Play out a set amount of games between two models.
    The score is the average game score, with 1 meaning that model1 won all games.

    Input:
        model1, model2 - pretrained PositionEvaluator model
        max_length - max length of a game
        num_games - amount of games to simulate
    Output:
        average_game_value - [-1,1] indicating average game score
    """
    tot_game_value = 0.
    for i in range(num_games):
        position = draughts.load_position()
        game_over = False
        win_value = 0
        subtree = None
        turns = 0
        while win_value == 0 and turns < max_length:
            turns += 1
            moves_dict = dict(draughts.possible_moves(*position))
            if len(moves_dict.keys()) == 0:
                break

            #Choose model, on odd games, model1 is white, on even games, model2 is white
            model = [model1,model2][int(position[1]) == i % 2]

            move, search_probabilities, subtree = MCTS(model, subtree, *position)

            position = (moves_dict[move], not position[1])
            win_value = draughts.check_win(position[0])

        if win_value == 0 and turns < max_length:
            win_value = int(position[1]) * -2 + 1

        if i % 2 == 1:
            tot_game_value -= win_value
        else:
            tot_game_value += win_value
    return tot_game_value/num_games

class GameStateDataset(Dataset):
    """GameStateDataset"""
    def __init__(self, data):
        """
        Args:
            data (list): List with gamestate data as tuples.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
