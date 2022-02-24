#Glob
from enum import Enum
class Pieces(Enum):
    EMPTY = 0
    WHITE_PIECE = 1
    WHITE_KING = 2
    BLACK_PIECE = 3
    BLACK_KING = 4

def coor_to_pos(h, w):
    return 5*h+w+1

def notation_move(h1,w1,h2,w2):
    return "{}-{}".format(coor_to_pos(h1,w1), coor_to_pos(h2,w2))
