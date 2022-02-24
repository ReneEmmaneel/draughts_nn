#Main goal of the

import torch

from utils import *

def possible_moves(board_state, is_white):
    """Given a board_state, find all possible moves and return
    a list of (move_name, new_board_state) tuples of all moves.

    input:
        board_state: tensor of size [h,w,p]
        is_white: true if it is whites move

    output:
        possible_moves: list of (move_name, new_board_state) tuples
    """
    possible_moves = []
    def add_move(move):
        new = board_state.detach().clone()
        new[h][w][Pieces.WHITE_PIECE.value] = 0
        new[h][w][Pieces.EMPTY.value] = 1
        new[move[0]][move[1]][Pieces.EMPTY.value] = 0
        new[move[0]][move[1]][Pieces.WHITE_PIECE.value] = 1
        notation = notation_move(h, w, move[0], move[1])
        possible_moves.append((notation, board_state))
    #Check captures

    #Check normal moves
    for h, row in enumerate(board_state):
        right = h % 2 == 0
        for w, tile in enumerate(row):
            if is_white:
                if h > 0:
                    #Check moves for normal pieces
                    if tile[Pieces.WHITE_PIECE.value] == 1:
                        move_left = (h-1, w - int(not right))
                        move_right = (h-1, w - int(not right) + 1)

                        for move in [move_left, move_right]:
                            if move[0] in range(10) and move[1] in range(5):
                                if board_state[move[0]][move[1]][Pieces.EMPTY.value]:
                                    add_move(move)
    return possible_moves


def print_position(board_state, is_white):
    print('==={} to move==='.format('white' if is_white else 'black'))
    for h, row in enumerate(board_state):
        if h % 2 == 0: print('  ', end='')
        for w, tile in enumerate(row):
            if tile[Pieces.WHITE_PIECE.value] == 1:
                print('w ', end='')
            elif tile[Pieces.WHITE_KING.value] == 1:
                print('W ', end='')
            elif tile[Pieces.BLACK_PIECE.value] == 1:
                print('b ', end='')
            elif tile[Pieces.BLACK_KING.value] == 1:
                print('B ', end='')
            else:
                print('# ', end='')

            if not (w == 4 and not h % 2 == 0):
                print('  ', end='')
        print('')

def load_position(FEN="W:W31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50:B1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
    turn, white, black = FEN.split(':')
    is_white = turn == 'W'
    white_pieces = white[1:].split(',')
    black_pieces = black[1:].split(',')

    board_state = torch.zeros(10, 5, 5)

    position = 1
    for row in board_state:
        for tile in row:
            if str(position) in white_pieces:
                tile[Pieces.WHITE_PIECE.value] = 1
            elif 'K' + str(position) in white_pieces:
                tile[Pieces.WHITE_KING.value] = 1
            elif str(position) in black_pieces:
                tile[Pieces.BLACK_PIECE.value] = 1
            elif 'K' + str(position) in black_pieces:
                tile[Pieces.BLACK_KING.value] = 1
            else:
                tile[Pieces.EMPTY.value] = 1
            position += 1
    return (board_state, is_white)

if __name__ == "__main__":
    position = load_position()
    print_position(*position)
    print([_[0] for _ in possible_moves(*position)])
