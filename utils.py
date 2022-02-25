#Glob
from enum import Enum
import torch
class Pieces(Enum):
    EMPTY = 0
    WHITE_PIECE = 1
    WHITE_KING = 2
    BLACK_PIECE = 3
    BLACK_KING = 4
    WHITE_PIECES = [1,2]
    BLACK_PIECES = [3,4]
    KINGS_PIECES = [2,4]

def coor_to_pos(h, w):
    return 5*h+w+1

def notation_move(h1,w1,h2,w2):
    return "{}-{}".format(coor_to_pos(h1,w1), coor_to_pos(h2,w2))

def notation_capture(fields):
    """field is list of tuples with coordinates"""
    return 'x'.join([str(coor_to_pos(f[0], f[1])) for f in fields])

def tensor_tile(piece):
    return torch.Tensor([1 if i == piece else 0 for i in range(5)])
