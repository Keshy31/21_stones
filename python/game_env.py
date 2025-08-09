import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StoneGameEnv(gym.Env):
    """
    Custom Gymnasium Environment for the 21 Stone Game.

    The game works as follows:
    - There are 21 stones initially.
    - Two players take turns removing 1, 2, or 3 stones.
    - The player who takes the last stone wins.

    In this environment, the agent plays against a computer-controlled opponent.

    ### Observation Space
    The observation is the number of stones remaining.
    - Type: Discrete(22) -> integers from 0 to 21.

    ### Action Space
    The agent can choose to take 1, 2, or 3 stones.
    - Type: Discrete(3)
      - 0: Take 1 stone
      - 1: Take 2 stones
      - 2: Take 3 stones

    ### Reward
    - +1: If the agent wins (takes the last stone).
    - -1: If the agent loses (opponent takes the last stone).
    -  0: For any other move during the game.

    ### Episode End
    An episode (one full game) ends when the number of stones reaches 0.
    """
    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self):
        """
        Initializes the game environment.

        This is where we define the core properties of the game, such as
        the action space and observation space.
        """
        super(StoneGameEnv, self).__init__()

        # Define the number of stones at the start of the game.
        self.initial_stones = 21

        # --- Action Space ---
        # The agent can take 1, 2, or 3 stones. This gives us 3 discrete actions.
        # We use `spaces.Discrete(3)` which creates a space of 3 possible integer values {0, 1, 2}.
        # In our `step` function, we will map these to taking 1, 2, and 3 stones.
        # The `action_space` is a required attribute for any Gymnasium environment.
        self.action_space = spaces.Discrete(3)

        # --- Observation Space ---
        # The state of the game is the number of stones remaining.
        # The number of stones can be any integer from 0 (game over) to 21.
        # This gives us 22 possible states.
        # We use `spaces.Discrete(22)` to represent these states.
        # The `observation_space` is also a required attribute.
        self.observation_space = spaces.Discrete(self.initial_stones + 1)

        # Initialize the state. This will be set properly in `reset()`.
        self.stones_remaining = self.initial_stones
        self.turn = 0 # 0 for agent, 1 for opponent

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state for a new episode.

        This method is called at the beginning of every new game. It sets the
        number of stones back to 21.

        Returns:
            observation (int): The initial state of the environment (21).
            info (dict): An empty dictionary, as we don't need auxiliary data here.
        """
        # We need to call the superclass's reset method for compatibility.
        super().reset(seed=seed)

        # Reset the number of stones to the initial count.
        self.stones_remaining = self.initial_stones
        
        # Randomly decide who starts
        self.turn = np.random.randint(0, 2)

        # If opponent starts, let them make a move.
        if self.turn == 1:
            self.opponent_move()

        # The initial observation is the starting number of stones.
        observation = self.stones_remaining
        # `info` can be used to pass auxiliary diagnostic information. We don't need it.
        info = {}

        return observation, info

    def step(self, action):
        """
        Executes one time step in the environment for the agent.
        """
        # The action from the agent is 0, 1, or 2. We map this to taking 1, 2, or 3 stones.
        stones_to_take = action + 1

        # --- Agent's Move ---
        if self.turn != 0:
            # This should not happen if the game logic is correct.
            # It's the opponent's turn, but step() was called.
            # We can return the current state without changes or handle as an error.
            return self.stones_remaining, 0, False, False, {"error": "Not agent's turn"}

        if stones_to_take > self.stones_remaining:
            self.stones_remaining = 0
            reward = -10.0 # Heavy penalty
            terminated = True
            observation = self.stones_remaining
            return observation, reward, terminated, False, {}

        self.stones_remaining -= stones_to_take

        if self.stones_remaining == 0:
            reward = 1.0  # Agent won
            terminated = True
            observation = self.stones_remaining
            return observation, reward, terminated, False, {}

        # It's now the opponent's turn
        self.turn = 1
        
        # Opponent makes their move
        self.opponent_move()
        
        observation = self.stones_remaining
        
        if self.stones_remaining == 0:
            reward = -1.0 # Opponent won
            terminated = True
        else:
            reward = 0.0
            terminated = False
            self.turn = 0 # Back to agent's turn

        return observation, reward, terminated, False, {}
        
    def opponent_move(self):
        # --- Opponent's Move ---
        # The optimal strategy is to always leave a number of stones that is a
        # multiple of 4 for the agent.
        opponent_stones_to_take = self.stones_remaining % 4
        
        # If the number of remaining stones is already a multiple of 4, the
        # opponent cannot force a win on this move, so it makes a random move.
        if opponent_stones_to_take == 0:
            opponent_stones_to_take = np.random.randint(1, 4)

        # Ensure the opponent's move is valid.
        opponent_stones_to_take = min(opponent_stones_to_take, self.stones_remaining)
        self.stones_remaining -= opponent_stones_to_take

    def render(self):
        """
        Renders the environment for human viewing.
        """
        if self.render_mode == 'human':
            print(f"Stones remaining: {self.stones_remaining}")

    def close(self):
        """
        Cleans up any resources used by the environment.
        """
        pass

