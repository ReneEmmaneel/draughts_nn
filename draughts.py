#Encodes the rules of international draughts.
#Main usage:
#   possible_moves(board_state, is_white) :-
#       returns a list of (notation, board_state) pairs
#   check_win(board_state) :- 0 if not a win, 1 if white win, -1 if black win
#
#There is also possibilties to play on command line when running this file
#
#TODO: encode drawing rules:
#The game is considered a draw when the same position repeats itself for the third time (not necessarily consecutive), with the same player having the move each time.
#A king-versus-king endgame is automatically declared a draw, as is any other position proven to be a draw.[citation needed]
#If, during 25 moves, there were only king movements, without piece movements or jumps, the game is considered a draw.
#If there are only three kings, two kings and a piece, or a king and two pieces against a king, the game will be considered a draw after the two players have each played 16 turns.
#Before a proposal for a draw can be made, at least 40 moves must have been made by each player.

import torch
import random

from utils import *

def check_win(board_state):
    """Given a board_state, returns 0 if not a win,
    1 if white has won, -1 if black has won
    """
    has_white = False
    has_black = False

    for row in board_state:
        for tile in row:
            if not has_white and torch.argmax(tile) in Pieces.WHITE_PIECES.value:
                has_white = True
            if not has_black and torch.argmax(tile) in Pieces.BLACK_PIECES.value:
                has_black = True
    if not has_black:
        return 1
    elif not has_white:
        return -1
    else:
        return 0

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
    longest_capture = 0

    def add_move(move, new_piece):
        """Using the current values h and w,
        and the input variable move which is a tuple,
        and a value to indicate which new_piece should be in the new positions,
        add a new board_state to possible_moves, including notation.
        """
        new = board_state.detach().clone()
        new[h][w] = tensor_tile(Pieces.EMPTY.value)
        if move[0] == 0 and new_piece == Pieces.WHITE_PIECE:
            new_piece = Pieces.WHITE_KING
        elif move[0] == 9 and new_piece == Pieces.BLACK_PIECE:
            new_piece = Pieces.BLACK_KING
        new[move[0]][move[1]] = tensor_tile(new_piece.value)
        notation = notation_move(h, w, move[0], move[1])
        possible_moves.append((notation, new))

    def continue_capture(h,w,board_state,captured_pieces=[], visited_fields=[]):
        """Recursive function to loop through all possible capturing sequences.
        It only keeps the longest sequence(s)
        """
        nonlocal longest_capture
        nonlocal possible_moves

        piece = torch.argmax(board_state[h][w]).item()
        if piece == Pieces.EMPTY.value:
            return
        elif piece in Pieces.WHITE_PIECES.value:
            opp_pieces = Pieces.BLACK_PIECES.value
        else:
            opp_pieces = Pieces.WHITE_PIECES.value
        is_king = piece in Pieces.KINGS_PIECES.value

        directions = [(-1,-1),(-1,0),(1,-1),(1,0)] #assuming left-aligned

        right = h % 2 == 0
        found = 0
        for dir in directions:
            if not is_king:
                move = (h+dir[0], w+dir[1] + int(right))
                if move in captured_pieces:
                    continue
                if move[0] in range(10) and move[1] in range(5):
                    if torch.argmax(board_state[move[0]][move[1]]).item() in opp_pieces:
                        capture = (move[0]+dir[0], move[1] + dir[1] + int(move[0] % 2 == 0))
                        if capture[0] in range(10) and capture[1] in range(5):
                            if board_state[capture[0]][capture[1]][Pieces.EMPTY.value] == 1:
                                found += 1
                                new = board_state.detach().clone()
                                new[h][w] = tensor_tile(Pieces.EMPTY.value)
                                new[capture[0]][capture[1]] = tensor_tile(piece)
                                continue_capture(capture[0], capture[1], new, captured_pieces=captured_pieces+[(move[0], move[1])], visited_fields=visited_fields+[(h,w)])
            else:
                move = (h, w)
                captured_piece = None
                for i in range(10):
                    move = (move[0]+dir[0], move[1] + dir[1] + int(move[0] % 2 == 0))
                    if move in captured_pieces:
                        break
                    if not (move[0] in range(10) and move[1] in range(5)):
                        break

                    if captured_piece == None:
                        if torch.argmax(board_state[move[0]][move[1]]).item() == Pieces.EMPTY.value:
                            continue
                        elif torch.argmax(board_state[move[0]][move[1]]).item() in opp_pieces:
                            captured_piece = move
                        else:
                            break
                    else:
                        if torch.argmax(board_state[move[0]][move[1]]).item() == Pieces.EMPTY.value:
                            found += 1
                            new = board_state.detach().clone()
                            new[h][w] = tensor_tile(Pieces.EMPTY.value)
                            new[move[0]][move[1]] = tensor_tile(piece)
                            continue_capture(move[0], move[1], new, captured_pieces=captured_pieces+[captured_piece], visited_fields=visited_fields+[(h,w)])
                        else:
                            break
        if found == 0 and len(captured_pieces) > 0:
            if len(captured_pieces) < longest_capture:
                return
            elif len(captured_pieces) > longest_capture:
                longest_capture = len(captured_pieces)
                possible_moves = []

            for captured_piece in captured_pieces:
                board_state[captured_piece[0]][captured_piece[1]] = tensor_tile(Pieces.EMPTY.value)

            if h == 0 and board_state[h][w][Pieces.WHITE_PIECE.value] == 1:
                board_state[h][w] = tensor_tile(Pieces.WHITE_KING.value)
            elif h == 9 and board_state[h][w][Pieces.BLACK_PIECE.value] == 1:
                board_state[h][w] = tensor_tile(Pieces.BLACK_KING.value)
            notation = notation_capture(visited_fields + [(h,w)])
            possible_moves.append((notation, board_state))

    #Check captures
    for h, row in enumerate(board_state):
        for w, tile in enumerate(row):
            if is_white and torch.argmax(board_state[h][w]) in Pieces.WHITE_PIECES.value:
               continue_capture(h,w,board_state)
            if (not is_white) and torch.argmax(board_state[h][w]) in Pieces.BLACK_PIECES.value:
               continue_capture(h,w,board_state)

    if longest_capture > 0:
        return possible_moves

    #Check normal moves
    for h, row in enumerate(board_state):
        right = h % 2 == 0
        for w, tile in enumerate(row):
            #Check normal move
            piece = Pieces.WHITE_PIECE if is_white else Pieces.BLACK_PIECE
            new_h = h-1 if is_white else h+1
            if tile[piece.value] == 1:
                move_left = (new_h, w - int(not right))
                move_right = (new_h, w - int(not right) + 1)

                for move in [move_left, move_right]:
                    if move[0] in range(10) and move[1] in range(5):
                        if board_state[move[0]][move[1]][Pieces.EMPTY.value]:
                            add_move(move, piece)

            #Check king move
            piece = Pieces.WHITE_KING if is_white else Pieces.BLACK_KING
            if tile[piece.value] == 1:
                new_h_up = h
                new_h_down = h
                new_w_left = w
                new_w_right = w

                moves_still_possible = [True, True, True, True]
                for _ in range(1,10):
                    new_h_up = new_h_up - 1
                    new_h_down = new_h_down + 1
                    new_w_left = new_w_left - int(new_h_up % 2 == 0)
                    new_w_right = new_w_right - int(new_h_up % 2 == 0) + 1
                    all_moves = [(new_h_up, new_w_left), (new_h_up, new_w_right),
                                 (new_h_down, new_w_left), (new_h_down, new_w_right)]

                    for move_num, move in enumerate(all_moves):
                        if moves_still_possible[move_num]:
                            if move[0] in range(10) and move[1] in range(5):
                                if board_state[move[0]][move[1]][Pieces.EMPTY.value]:
                                    add_move(move, piece)
                                else:
                                    moves_still_possible[move_num] = False
                            else:
                                moves_still_possible[move_num] = False
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
                print('. ', end='')

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

def play_game():
    position = load_position()
    print_position(*position)

    game_over = False
    win_value = 0
    while win_value == 0:
        moves_dict = dict(possible_moves(*position))
        if len(moves_dict.keys()) == 0:
            break
        while True:
            move = input("Enter move: ")
            if move == 'r':
                move = random.choice(list(moves_dict.keys()))
                print('Playing {}'.format(move))
            elif not move in moves_dict.keys():
                print('Not a valid move')
                print(list(moves_dict.keys()))
                continue
            position = (moves_dict[move], not position[1])
            print_position(*position)
            break
        win_value = check_win(position[0])

    if win_value == 0:
        win_value = int(position[1]) * -2 + 1
        print(int(position[1]))
    if win_value == 1:
        print('White won the game!')
    elif win_value == -1:
        print('Black won the game!')


if __name__ == "__main__":
    play_game()
