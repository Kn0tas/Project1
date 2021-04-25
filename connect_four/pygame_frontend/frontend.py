import pygame
import time
import numpy as np

from connect_four.backend.game import agent_by_mark, ConnectFourEnv, next_mark # possibly need to set export DISPLAY=:0 in terminal?
from connect_four.backend.play_human import HumanAgent


### GLOABAL PARAMS ####

# Define parameters
NUM_ROWS = 6
NUM_COLS = 7

CELL_RADIUS = 20
CELL_PADDING = 15

BOARD_HEIGHT = NUM_ROWS * (2*CELL_RADIUS + CELL_PADDING) + CELL_PADDING
BOARD_WIDTH = NUM_COLS * (2*CELL_RADIUS + CELL_PADDING) + CELL_PADDING

#VELOCITY = 1
MARKER_RADIUS = CELL_RADIUS

BOARD_COLOR = pygame.Color('blue')
CELL_COLOR = pygame.Color('black')
MARKER_1_COLOR = pygame.Color('yellow')
MARKER_2_COLOR = pygame.Color('red')

DROP_PANEL_WIDTH = BOARD_WIDTH
DROP_PANEL_HEIGHT = 80
DROP_PANEL_COLOR = pygame.Color('black')

SCREEN_WIDTH = BOARD_WIDTH
SCREEN_HEIGHT = BOARD_HEIGHT + DROP_PANEL_HEIGHT


# draw empty cells
def get_pixel_map():
    global screen
    global CELL_RADIUS, CELL_PADDING, CELL_COLOR
    global NUM_ROWS, NUM_COLS
    
    IDXS_TO_PIXEL_CORDS_MAP = dict()

    y_pos = DROP_PANEL_HEIGHT + CELL_RADIUS + CELL_PADDING
    for row_idx in range(NUM_ROWS):
        x_pos = CELL_RADIUS + CELL_PADDING
        for col_idx in range(NUM_COLS):
            IDXS_TO_PIXEL_CORDS_MAP[(row_idx, col_idx)] = (x_pos, y_pos)
            pygame.draw.circle(screen, CELL_COLOR, (x_pos, y_pos), CELL_RADIUS)
            x_pos += 2 * CELL_RADIUS + CELL_PADDING
        y_pos += 2 * CELL_RADIUS + CELL_PADDING
    
    return IDXS_TO_PIXEL_CORDS_MAP

def draw_board(board):
    global IDXS_TO_PIXEL_CORDS_MAP
    
    if isinstance(board, tuple): # undo board flattening
        board = np.array(board).reshape(6, 7)
    else:
        board = board.reshape(6,7)
        
    color_map = {0: CELL_COLOR, -1: MARKER_1_COLOR, 1: MARKER_2_COLOR}
    
    for row_index in range(board.shape[0]):
        for col_index in range(board.shape[1]):            
            marker_value = board[row_index, col_index]
            marker_color = color_map[int(marker_value)]
            pygame.draw.circle(screen, marker_color, IDXS_TO_PIXEL_CORDS_MAP[(row_index, col_index)], CELL_RADIUS)


def init_pygame_backend():

    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) # returns the surface where you can draw stuff
    pygame.draw.rect(screen, DROP_PANEL_COLOR, pygame.Rect((0,0), (DROP_PANEL_WIDTH, DROP_PANEL_HEIGHT)))
    pygame.draw.rect(screen, BOARD_COLOR, pygame.Rect((0,DROP_PANEL_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT)))

    IDXS_TO_PIXEL_CORDS_MAP = get_pixel_map()  # make global?
    #pygame.display.flip()

    draw_board(board)

if __name__ == '__main__':

    
    #human_agent = HumanAgent('O')
    #ai_agent = ConnectFourA2C.load('a2c_agent_50k_vs100kA2C')
        
    state = env.reset()
    board, mark = state
    done = False

    init_pygame_backend(board)





    