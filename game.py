#In this file the bot opponent is defined.

import logging
import gym
import numpy as np

from gym import spaces

CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
NUM_LOC = 9
O_REWARD = 1
X_REWARD = -1
NO_REWARD = 0
NUM_ROWS = 6
NUM_COLUMNS = 7

LEFT_PAD = '  '
LOG_FMT = logging.Formatter('%(levelname)s '
                            '[%(filename)s:%(lineno)d] %(message)s',
                            '%Y-%m-%d %H:%M:%S')

def tomark(code):
    return CODE_MARK_MAP[code]

def tocode(mark):
    return 1 if mark == 'O' else 2

def next_mark(mark):
    return 'X' if mark == 'O' else 'O'

def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent

def after_action_state(state, action):
    """Execute an action and returns resulted state.
    Args:
        state (tuple): Board status + mark
        action (int): Action to run
    Returns:
        tuple: New state
    """

    board, mark = state
    nboard = list(board[:])
    nboard[action] = tocode(mark)
    nboard = tuple(nboard)
    return nboard, next_mark(mark)

### GAME STATUS SECTION ###

class MarkerCounter:
    def __init__(self):
        self.maker_count = 0
        self.previous_marker = None

    def reset(self):
        self.maker_count = 0
        self.previous_marker = None
    
    def count(self, cell_val):
        # marker_val: 0 (empty) or 1 or 2
        if self.previous_marker != cell_val:
            self.reset()
        if cell_val in [1,2]:

            self.maker_count += 1
            self.previous_marker = cell_val
            
        if self.maker_count == 4:
            return cell_val
        else:
            return None

def check_non_diagonal_winner(board):
    marker_counter = MarkerCounter()

    for row in board:
        marker_counter.reset()
        for cell in row:
            gameover_flag = marker_counter.count(cell)
            if gameover_flag in [1, 2]:
                print(f'The winner is player: {gameover_flag}')
                return gameover_flag
            
    for col in board.T:
        marker_counter.reset()
        for cell in col:
            gameover_flag = marker_counter.count(cell)
            if gameover_flag in [1, 2]:
                print(f'The winner is player: {gameover_flag}')
                return gameover_flag
    return None
        
def check_diagonal_winner(board):
    marker_counter = MarkerCounter()
    
    down_right_startpos = [(2,0), (1,0), (0,0), (0,1), (0,2), (0,3)]    
    for dr_start_pos in down_right_startpos:
        marker_counter.reset()
        row_i, col_j = dr_start_pos
        while row_i < NUM_ROWS and col_j < NUM_COLUMNS:
            cell = board[row_i, col_j]
            gameover_flag = marker_counter.count(cell)
            #print(row_i, col_j, cell, gameover_flag)
            if gameover_flag in [1, 2]:
                print(f'The winner is player: {gameover_flag}')
                return gameover_flag
            row_i += 1
            col_j += 1
            
    up_right_startpos = [(3,0), (4,0), (5,0), (5,1), (5,2), (5,3)]
    for ur_start_pos in up_right_startpos:
        marker_counter.reset()
        row_i, col_j = ur_start_pos
        while -1 < row_i and col_j < NUM_COLUMNS:
            cell = board[row_i, col_j]
            gameover_flag = marker_counter.count(cell)
            print(row_i, col_j, cell, gameover_flag)
            if gameover_flag in [1, 2]:
                print(f'The winner is player: {gameover_flag}')
                return gameover_flag
            row_i -= 1
            col_j += 1

    return None
    
def check_game_draw(board):
    return ~(board == 0).any()
    
def check_game_status(board):
    """Return game status by current board status.
    Args:
        board (list): Current board state
    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).
    """
    if isinstance(board, tuple):
        board = np.array(board).reshape(6, 7)[::-1, :]

    if check_game_draw(board):
        return 0 # game draw
        
    gameover_flag = check_non_diagonal_winner(board)
    if gameover_flag is None:
        gameover_flag = check_diagonal_winner(board)
    
    if gameover_flag is not None:
        return gameover_flag # winner is 1 or 2
    else:
        return -1 # game still ongoing
    
### GAME STATUS SECTION END ###


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=0.02, show_number=False):
        self.action_space = spaces.Discrete(NUM_COLUMNS)
        self.observation_space = spaces.Discrete(NUM_COLUMNS * NUM_ROWS)
        self.alpha = alpha
        self.set_start_mark('O')
        self.show_number = show_number
        self.seed()
        self.reset()

    def set_start_mark(self, mark):
        self.start_mark = mark

    def reset(self):
        self.board = np.zeros((NUM_ROWS,NUM_COLUMNS)) # make this a board of 0. (Game Start)
        self.mark = self.start_mark
        self.done = False
        return self._get_obs()

    def step(self, action):
        """Step environment by action.
        Args:
            action (int): Location
        Returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """

        assert self.action_space.contains(action)
        assert self.board[:, action][0] == 0           # Check that the column is not full.

        loc = action
        if self.done:
            return self._get_obs(), 0, True, None

        reward = NO_REWARD
        # place
        col_vals = self.board[:, loc] # get the column
        row_index = np.argwhere(col_vals == 0)[-1, -1] # get index of "bottom-most" empty cell
        self.board[row_index, loc] = tocode(self.mark)

        # check game status
        status = check_game_status(self.board)
        logging.debug("check_game_status board {} mark '{}'"
                      " status {}".format(self.board, self.mark, status))
        if status >= 0:
            self.done = True
            if status in [1, 2]:
                # always called by self
                reward = O_REWARD if self.mark == 'O' else X_REWARD

        # switch turn
        self.mark = next_mark(self.mark)
        return self._get_obs(), reward, self.done, None

    def _get_obs(self):
        # TODO: "unroll array to list!"
        #return self.board.flatten(), self.mark
        return tuple(self.board.flatten()), self.mark

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_board(print)  # NOQA
            print('')
        else:
            self._show_board(logging.info)
            logging.info('')

    def show_episode(self, human, episode):
        self._show_episode(print if human else logging.warning, episode)

    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))

    def _show_board(self, showfn):
        """Draw tictactoe board."""
        board2disp = self.board.astype(object)
        for numval, symbval in CODE_MARK_MAP.items():
            board2disp[self.board == numval] = symbval 
        showfn(board2disp)

    def show_turn(self, human, mark):
        self._show_turn(print if human else logging.info, mark)

    def _show_turn(self, showfn, mark):
        showfn("{}'s turn.".format(mark))

    def show_result(self, human, mark, reward):
        self._show_result(print if human else logging.info, mark, reward)

    def _show_result(self, showfn, mark, reward):
        status = check_game_status(self.board)
        assert status >= 0
        if status == 0:
            showfn("==== Finished: Draw ====")
        else:
            msg = "Winner is '{}'!".format(tomark(status))
            showfn("==== Finished: {} ====".format(msg))
        showfn('')

    def available_actions(self):
        available_actions = [action_index for action_index, cell_val in enumerate(self.board[0, :]) if cell_val == 0.0]
        return available_actions

def set_log_level_by(verbosity):
    """Set log level by verbosity level.
    verbosity vs log level:
        0 -> logging.ERROR
        1 -> logging.WARNING
        2 -> logging.INFO
        3 -> logging.DEBUG
    Args:
        verbosity (int): Verbosity level given by CLI option.
    Returns:
        (int): Matching log level.
    """
    if verbosity == 0:
        level = 40
    elif verbosity == 1:
        level = 30
    elif verbosity == 2:
        level = 20
    elif verbosity >= 3:
        level = 10

    logger = logging.getLogger()
    logger.setLevel(level)
    if len(logger.handlers):
        handler = logger.handlers[0]
    else:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(LOG_FMT)
    return level

##############

if __name__ == '__main__':
    env = TicTacToeEnv()
    env.reset()
    env.render()

#for _ in range(1000):
#   env.render()
#    env.step(env.action_space.sample()) # take a random action
#env.close()