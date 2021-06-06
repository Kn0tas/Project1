import random
from stable_baselines3.common import logger
from collections import deque

from connect_four.backend.game import ConnectFourEnv
#from connect_four.backend.play_human import HumanAgent

from stable_baselines3.a2c import A2C
from gym import spaces
from torch.nn import functional as F


import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv

from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance

# config params...
MAX_PROB_ACTION_SAMPLES = 10
#EPSILON = 0.2

class ConnectFourOnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        epsilon = 0.0
    ):

        super(ConnectFourOnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.epsilon = epsilon

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        def do_early_exit_due_to_winning(ep_info_buffer, iteration, force = False):
            # TODO: check frac wins, exit if needed
            if force or ((len(ep_info_buffer) == ep_info_buffer.maxlen) and iteration % 250):
                # TODO: make this block smarter
                wins_in_queue = [0 < d['r'] for d in ep_info_buffer]
                fraction_of_wins = np.mean(wins_in_queue)
                print("fraction_of_wins after {iteration} iterations: ", fraction_of_wins)
                if 0.9 < fraction_of_wins:
                    print(f"Exiting due to more than 90% wins in the last 100 games after {iteration} iterations!")
                    print(f'Agent total updates: {agent_to_improve._n_updates}')
                    return True
                else:
                    return False

        print(self.env)
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

                if do_early_exit_due_to_winning(self.ep_info_buffer, iteration):
                    break

            self.train()

        callback.on_training_end()
        print('Exiting training due to max timesteps reached!')
        _ = do_early_exit_due_to_winning(self.ep_info_buffer, iteration, force = True)
        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():

                assert len(env.envs) == 1 # currently does not support multieple envs..

                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                
                sample_count = 0
                while True:
                    sample_count += 1
                    actions, values, log_probs = self.policy.forward(obs_tensor, deterministic = False)
                    agent_action_cached = actions
                    
                    # infer epsilon from environment...? (TODO: maybe infer from somewhere else)
                    epsilon = self.epsilon # env.envs[0].epsilon or -99
                    #print("epsilon ", epsilon)
                    if (np.random.rand() < epsilon) or (MAX_PROB_ACTION_SAMPLES <= sample_count):
                        
                        all_actions = [0,1,2,3,4,5,6]
                        valid_actions = [a for a in all_actions if env.envs[0].check_action_valid(a)]
                        actions = np.array([random.choice(valid_actions)])
                        #print(f'RADNOM ACTION TRIGGER! {agent_action_cached.cpu().numpy()[0]} --> {actions[0]}')
                        break
                    
                    if env.envs[0].check_action_valid(actions.cpu().numpy()[0]):
                        actions = actions.cpu().numpy()
                        break

                #print('from collect_rollouts, sample count was: ', sample_count)
                
                """
                for potential_action in all_actions:
                    if env.envs[0].check_action_valid(potential_action): # TODO: implement env-is-valid check!
                        obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                        actions, values, log_probs = self.policy.forward(obs_tensor, deterministic = False)
                        

                        print(log_probs)
                        print(actions)
                        assert False, 'delib failure!'
                        #print(log_probs)
                        action_probabilities[potential_action] = log_probs.cpu().numpy()[0] # why 0th index?

                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)

                # tmp debug
                log_probs_arr = log_probs.cpu().numpy()
                print(log_probs)
                print(log_probs_arr)
                if np.unique(log_probs_arr).shape[0] != log_probs_arr.shape[0]:
                    print('action probs has dups!')
                    print(log_probs_arr)

                #print('from collect_rollouts: ', action_probabilities)
                if 0.01 > np.random.rand():
                    assert False, 'delib failure!'

            # OBS: Will always select first valid aciton if all probs are the same ! # TODO: sample instead !
            MOST_PROBABLE_VALID_ACTION = max([(k, v) for k, v in action_probabilities.items()], key = lambda x: x[1])[0]

            #print(action_probabilities, MOST_PROBABLE_VALID_ACTION)
            #print('')
            #print(actions, values, log_probs)
            
            #actions = actions.cpu().numpy()
            #print(actions, type(actions))

            #print('----')

            actions = np.array([MOST_PROBABLE_VALID_ACTION])
            #print(actions, type(actions))


            #print(action_probabilities)
            """


            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)


            #print('HEJHEJ')
            #print(clipped_actions)
            #valid_actions = env.check_valid_actions()
            
            #action_score = [0.1, 0.2, 0.3, 0.0 ,-0.2]

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            #print(rewards)
            #assert False, 'delib crash'

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

class ConnectFourA2C(ConnectFourOnPolicyAlgorithm):
    
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(ConnectFourA2C, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()


    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # TODO: avoid second computation of everything because of the gradient


            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/explained_variance", explained_var)
        logger.record("train/entropy_loss", entropy_loss.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())


    def predict_with_invalid_mask(self, observation, env = None, deterministic = False):
        """ Get most probable action that is valid wrt. env state.

        """

        if env is None:
            assert len(self.env.envs) == 1  # currently does not support multieple envs..
            env = self.env.envs[0]

        with th.no_grad():

            obs_tensor = th.as_tensor(self._last_obs).to(self.device)
            sample_count = 0
            while True:
                sample_count += 1
                actions, values, log_probs = self.policy.forward(obs_tensor, deterministic = deterministic) # TODO: Should be deteminsitc? & invalid filtered...?
                action = actions.cpu().numpy()[0]
                if env.check_action_valid(action):
                    break

                if MAX_PROB_ACTION_SAMPLES <= sample_count:
                    all_actions = [0,1,2,3,4,5,6]
                    valid_actions = [a for a in all_actions if env.check_action_valid(a)]
                    action = random.choice(valid_actions)
                    break

            #print('from predict_with_invalid mask, sample count was: ', sample_count)
            
            """

            # COMPUTE HIGHEST-PROB VALID ACTION!
            all_actions = [0,1,2,3,4,5,6] # TODO: Infer from env!
            action_probabilities = {}

            for potential_action in all_actions:
                if env.check_action_valid(potential_action):
                    obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                    actions, values, log_probs = self.policy.forward(obs_tensor)
                    #print(log_probs)
                    action_probabilities[potential_action] = log_probs.cpu().numpy()[0]


            print(action_probabilities)
            if 0.05 > np.random.rand():
                assert False, 'delib failure!'

            MOST_PROBABLE_VALID_ACTION = max([(k, v) for k, v in action_probabilities.items()], key = lambda x: x[1])[0]
            #actions = np.array([MOST_PROBABLE_VALID_ACTION])
        """
        return action
        #return self.policy.predict(observation, state, mask, deterministic)

# init SB agent...
# train SB-agent on connect4-env
# log performance.... every N iteration... (e.g. by playing 100 games with random-agent)

if __name__ == '__main__':

    #tf.logging.set_verbosity(tf.logging.ERROR)


    #env_ = ConnectFourEnv()
    #opponent_agent = ConnectFourA2C.load('a2c_agent_50k_vs_random') # 'a2c_agent_50k_vs100kA2C' #ConnectFourA2C.load('a2c_agent_100k')#, env = env_)
    
    #import os
    #print(os.listdir('..'))
    #assert False
    """
    opponent_agent = 'random'

    print(f'Using opponent_agent: {opponent_agent}')


    env = ConnectFourEnv(opponent = opponent_agent)

    #obs = env.reset()
    #print(obs, obs.shape)
    #print(env.observation_space)

    #from stable_baselines.common.env_checker import check_env
    #check_env(env)

    #print(env.observation_space)

    agent = ConnectFourA2C(policy = 'MlpPolicy', env=env, verbose=1, tensorboard_log="./test_tensorboard/")
    #agent2 = ConnectFourA2C(policy = 'MlpPolicy', env=env(opponent = agent1), verbose=1)


    total_timesteps = 10000
    agent.learn(total_timesteps=total_timesteps)
    agent_name = f'A2Cagent_epsilon{EPSILON}_{int(total_timesteps//1e3)}k_vs_{str(opponent_agent)}'
    print('SAVING AGENT: ' + agent_name)
    agent.save(agent_name)

    """

    import os
    def create_new_agent_generation(experiment_dir):

        def load_agents_from_dir(previous_generation_dir):
            dirpath = os.path.join(experiment_dir, previous_generation_dir)
            print(dirpath)
            files = os.listdir(dirpath)
            agents = [ConnectFourA2C.load(os.path.join(dirpath, f)) for f in files]
            return agents

        if experiment_dir is None:
            previous_agents_list = 'random'
            experiment_dir = 'experiment_runs' + str(int(time.time()))
            os.makedirs(experiment_dir) # create fold for agent generations
            generation_index = 1
            
        else:
            previous_generation_dir = max(os.listdir(experiment_dir))
            previous_agents_list = load_agents_from_dir(previous_generation_dir)
            generation_index = len((os.listdir(experiment_dir))) + 1

        agent_generation_dirname = 'generation_' + str(generation_index)

        """ 
        total_timesteps = 1000
        for epsilon in [0.05, 0.2, 0.5]:

            # create "environment" with previous agent-group as opponent
            if previous_agents_list == 'random':
                env = ConnectFourEnv(opponent = 'random')
            else:
                env = ConnectFourEnv(opponent_group = previous_agents_list, epsilon = epsilon)
            # create new agent with current epislon
            new_agent = ConnectFourA2C(policy = 'MlpPolicy', env=env, verbose=1)
            agent_name = f'A2Cagent_epsilon{epsilon}_{int(total_timesteps//1e3)}k_vs_generation_{generation_index-1}'
            # train the agent
            print("training agent: " + agent_name)
            new_agent.learn(total_timesteps=total_timesteps) # TODO: might want to train variable number of steps depending on results?
            # save the agent
            print("saving agent: " + agent_name + " to: " + os.path.join(experiment_dir, agent_generation_dirname, agent_name))

            agent_name = f'A2Cagent_epsilon{epsilon}_{int(total_timesteps//1e3)}k_vs_generation_{generation_index-1}'
            new_agent.save(os.path.join(experiment_dir, agent_generation_dirname, agent_name))

        """

    #previous_agents_list
    from copy import deepcopy
    import gc

    experiment_dirpath = None#'experiment_runs1622994637'

    for generation in range(5):

        if experiment_dirpath is None:

            experiment_dirpath = 'experiment_runs' + str(int(time.time()))
            os.makedirs(experiment_dirpath) # create fold for agent generations
            
            training_env = ConnectFourEnv(opponent = 'random')
            agent_to_improve = ConnectFourA2C(policy = 'MlpPolicy', env=training_env, verbose=1)

        else:
            # load previous agent
            #previous_agent_fname = max(os.listdir(experiment_dirpath))
            #print("Loading agent: ", previous_agent_fname)
            #agent_to_improve = ConnectFourA2C.load(os.path.join(experiment_dirpath, previous_agent_fname))

            opponent_agent_clones = [] # make clones with diffrent epislons
            for epsilon in [0.05, 0.2, 0.5]:
                opponent_clone = deepcopy(agent_to_improve)
                opponent_clone.epsilon = epsilon
                opponent_agent_clones.append(opponent_clone)
            print("Made opponent clones ", opponent_agent_clones)
            #print(agent_to_improve.env)
            training_env.opponent = None
            training_env.opponent_group = opponent_agent_clones
            #training_env = ConnectFourEnv(opponent_group = opponent_agent_clones)
            #agent_to_improve.env = training_env # update the env to include new opponents
            #agent_to_improve.env.num_envs = 1

        max_total_timesteps = 50000
        input("Press any key to init training")
        agent_to_improve.learn(total_timesteps=max_total_timesteps, log_interval = 100)
        # reset episode queue results prior to saving
        new_queue = agent_to_improve.ep_info_buffer = deque(maxlen=agent_to_improve.ep_info_buffer.maxlen)

        agent_name = f'A2C_agent_n_updates_{agent_to_improve._n_updates}'
        print('saving agent')
        agent_to_improve.save(os.path.join(experiment_dirpath, agent_name))
        #del agent_to_improve, training_env
        #gc.collect()

    # TODO, save agent!

    #create_new_agent_generation("experiment_runs1621633673")
    #print(agent)
    # TODO: randomize starting turn184
    