from game import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD
from play_human import HumanAgent

import tensorflow as tf

from stable_baselines.a2c import A2C
from stable_baselines.common.policies import MlpPolicy



# init SB agent...
# train SB-agent on connect4-env
# log performance.... every N iteration... (e.g. by playing 100 games with random-agent)

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.ERROR)
    env = TicTacToeEnv()
    #obs = env.reset()
    #print(obs, obs.shape)
    #print(env.observation_space)

    #from stable_baselines.common.env_checker import check_env
    #check_env(env)

    #print(env.observation_space)


    agent = A2C(policy = MlpPolicy, env=env, verbose=1)
    agent.learn(total_timesteps=1000)
    #print(agent)