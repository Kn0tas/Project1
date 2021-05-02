# Project1
Play connect four vs AI!

## Installation
- 1. Activate conda env (connect4) <-- TODO: add requirements-file instead!
- 2. cd to project root
- 3. run in terminal: python3 setup.py developV

## Train an AI agent

Cd to connect_four/backend, then open the train_sb_agent.py-script

Add opponent agent by either loading an existing agent by

```opponent_agent = ConnectFourA2C.load('../path/to/my/agent.zip')```

...or by using a random opponent by setting to a string 'random'.

Then set number of training episods by:

```agent.learn(total_timesteps=100000)```

Finally, set the filepath of the agent to be saved:

```agent.save('nameOfAgentToBeSaved') ```


## Play an AI agent in the frontend

Cd to connect_four/pygame_frontend, then from terminal, run:

```python3 frontend.py --player1_string=firstPlayerString --player2_string=secondPlayerString```

where valid player strings are; human, random or a filepath to an agent file.

__note__: Currently, possibly the agent file needs to be put in the same directory as the fronted.py-script in order for agent loading to work?


__note 2__: If using WSL, prior to launching the run_game-script from terminal, start Xming run:

```export DISPLAY=:0 ```
