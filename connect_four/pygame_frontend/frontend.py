import time
import random
import numpy as np
import pygame
import click

from connect_four.backend.game import agent_by_mark, ConnectFourEnv, next_mark # possibly need to set export DISPLAY=:0 in terminal?
from connect_four.backend.play_human import HumanAgent
from connect_four.backend.train_sb_agent import ConnectFourA2C
from connect_four.backend.base_agent import RandomAgent


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
    #global screen
    global IDXS_TO_PIXEL_CORDS_MAP
    #global CELL_RADIUS, CELL_PADDING, CELL_COLOR
    #global NUM_ROWS, NUM_COLS
    
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
    #global IDXS_TO_PIXEL_CORDS_MAP
    #global screen
    
    if isinstance(board, tuple): # undo board flattening
        board = np.array(board).reshape(6, 7)
    else:
        board = board.reshape(6,7)
        
    color_map = {0: CELL_COLOR, 1: MARKER_1_COLOR, -1: MARKER_2_COLOR}
    
    for row_index in range(board.shape[0]):
        for col_index in range(board.shape[1]):            
            marker_value = board[row_index, col_index]
            marker_color = color_map[int(marker_value)]
            pygame.draw.circle(screen, marker_color, IDXS_TO_PIXEL_CORDS_MAP[(row_index, col_index)], CELL_RADIUS)

def init_pygame_backend(board):

    global screen

    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) # returns the surface where you can draw stuff
    pygame.draw.rect(screen, DROP_PANEL_COLOR, pygame.Rect((0,0), (DROP_PANEL_WIDTH, DROP_PANEL_HEIGHT)))
    pygame.draw.rect(screen, BOARD_COLOR, pygame.Rect((0,DROP_PANEL_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT)))

    IDXS_TO_PIXEL_CORDS_MAP = get_pixel_map()  # make global?
    #pygame.display.flip()

    draw_board(board)

@click.command()
@click.option('--player1_string')#, help="Player 1 string.")
@click.option('--player2_string')#, help="Player 2 string.") 
def run_game(player1_string, player2_string):

    print(player1_string)
    print(player2_string) # ../agents/a2c_agent_50k_vs100kA2C # a2c_agent_50k_vs_random

    # setup playing agents
    def init_agent(player_string, marker):
        if player_string == 'human':
            player_name = 'A lousy human'
            agent = HumanAgent(marker)
        elif player_string == 'random':
            player_name = 'A very drunk robot'
            agent = RandomAgent(marker)
        else:
            player_name = 'The grinder bot-a-tron-3000'
            agent = ConnectFourA2C.load(player_string)
        return agent, player_name
    player1_agent, player1_name = init_agent(player1_string, 'O')
    player2_agent, player2_name = init_agent(player2_string, 'X')

    player1_name = 'bot-a-tron-5000'
    player2_name = 'bot-a-tron-3000'


    marker_to_agent_map = {
        'O': (player1_agent, player1_name, MARKER_1_COLOR), # yellow
        'X': (player2_agent, player2_name, MARKER_2_COLOR),
    }
    get_agent_by_mark = lambda marker: marker_to_agent_map[marker][0]
    get_player_name_by_mark = lambda marker: marker_to_agent_map[marker][1]
    get_player_color_by_mark = lambda marker: marker_to_agent_map[marker][2]

    # TODO: Draw the player names...
    #font = pygame.font.SysFont(None, 24)
    #img = font.render('WINNER: ' + winning_player_name, True, pygame.Color('green'))
    #screen.blit(img, (20, 20))


    get_agent_by_mark = lambda marker: marker_to_agent_map[marker][0]

    # setup environment
    env = ConnectFourEnv(show_number=False,
                         interactive_mode = True,
                         opponent = 'random')     # random??
    state = env.reset()
    observation, mark = state
    print(mark)
    mark = 'X'
    print(mark)
    done = False

    # init frondtend
    init_pygame_backend(observation)
    # draw player names...
    fontsize = 18
    font = pygame.font.SysFont(None, fontsize)
    img1 = font.render('Player 1: ' + player1_name, True, MARKER_1_COLOR)
    screen.blit(img1, (10, 10))
    img2 = font.render('Player 2: ' + player2_name, True, MARKER_2_COLOR)
    screen.blit(img2, (BOARD_WIDTH-180, 10))

    # enter playing loop
    try:
        pygame.display.flip()
        while True:
            e = pygame.event.poll()
            if e.type == pygame.QUIT:
                break

            pygame.display.flip()                

            current_player_mark = mark

            env.show_turn(True, mark) # print out which player to play
            agent2play = get_agent_by_mark(mark)

            if isinstance(agent2play, HumanAgent): # do mouse click action
                print('Human agent!!')
                
                while True:
                    e = pygame.event.poll()
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if e.type == pygame.MOUSEBUTTONDOWN:
                        print('detected mouse event!')
                        CLICK_ACTION = int(mouse_x // (SCREEN_WIDTH/NUM_COLS))
                        ava_actions = env.available_actions()
                        action, action_is_valid = agent2play.pygame_act(ava_actions,CLICK_ACTION)
                        if not action_is_valid:
                            print('Invalid action not supported, make a valid action!')
                        else:
                            break
            else: # do ai-agent action
                print('AI agent!!')
                if isinstance(agent2play, RandomAgent):
                    ava_actions = env.available_actions()
                    action = agent2play.act(ava_actions)
                else:
                    action = agent2play.predict_with_invalid_mask(observation, env = env, deterministic = False) # dont sample -- take highest prob always
                time.sleep(0.5 + 1.5*random.random()) # fake "thinking"
            
            # then; update env with collected action
            state, reward, done, info = env._step(action)
            observation, mark = state
            draw_board(observation)
            if done:
                env.show_result(True, mark, reward)


                #DROP_PANEL_WIDTH = BOARD_WIDTH
                #DROP_PANEL_HEIGHT = 80

                winning_player_name = get_player_name_by_mark(current_player_mark)
                fontsize = 30
                font = pygame.font.SysFont(None, fontsize)
                img = font.render('WINNER: ' + winning_player_name, True, get_player_color_by_mark(current_player_mark))
                screen.blit(img, (90, DROP_PANEL_HEIGHT/2 - fontsize/2 + 20)) # x y? + y margin

                pygame.display.flip()
                break

        print('Game over! Click closing symnbol to quit...')
        while True:
            e = pygame.event.poll()
            if e.type == pygame.QUIT:
                break

    except Exception as e:
        pygame.quit()
        raise e
    finally:
        pygame.quit()

if __name__ == '__main__':
    run_game()
