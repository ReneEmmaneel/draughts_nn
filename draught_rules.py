#Main goal of the

import torch

from utils import Pieces

def possible_moves(board_state, is_white):
    """Given a board_state, find all possible moves and return
    a list of (move_name, new_board_state) tuples of all moves.

    input:
        board_state: tensor of size [h,w,p]
        is_white: true if it is whites move

    output:
        possible_moves: list of (move_name, new_board_state) tuples
    """
    pass

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
    print_position(*load_position())
