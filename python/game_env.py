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

        # The initial observation is the starting number of stones.
        observation = self.stones_remaining
        # `info` can be used to pass auxiliary diagnostic information. We don't need it.
        info = {}

        return observation, info

    def step(self, action):
        """
        Executes one time step in the environment.

        This is the core of the environment. It takes an agent's action,
        updates the game state, simulates the opponent's move, and calculates the reward.

        Args:
            action (int): The action chosen by the agent (0, 1, or 2).

        Returns:
            A tuple containing:
            - observation (int): The number of stones remaining after the opponent's turn.
            - reward (float): The reward for the agent's move.
            - terminated (bool): True if the game is over, False otherwise.
            - truncated (bool): Always False, as our game has a clear end.
            - info (dict): An empty dictionary.
        """
        # The action from the agent is 0, 1, or 2. We map this to taking 1, 2, or 3 stones.
        stones_to_take = action + 1

        # --- Agent's Move ---
        # It's good practice to check if the move is valid.
        if stones_to_take > self.stones_remaining:
            # This is an illegal move. In a real game, this might happen if the AI
            # hasn't learned the rules yet. We'll penalize it heavily and end the game.
            self.stones_remaining = 0
            reward = -10.0 # Heavy penalty to discourage this
            terminated = True
            observation = self.stones_remaining
            return observation, reward, terminated, False, {}

        # Apply the agent's action.
        self.stones_remaining -= stones_to_take

        # Check if the agent won with this move.
        if self.stones_remaining == 0:
            reward = 1.0  # Agent took the last stone(s) and won.
            terminated = True
            observation = self.stones_remaining
            return observation, reward, terminated, False, {}

        # --- Opponent's Move ---
        # Now it's the opponent's turn. We'll implement a simple but effective strategy.
        # The optimal strategy in this game is to always leave a number of stones
        # that is a multiple of 4 for the other player.
        # Example: If there are 9 stones, taking 1 leaves 8 (a multiple of 4).
        opponent_stones_to_take = (self.stones_remaining - 1) % 4
        if opponent_stones_to_take == 0:
            # If the opponent cannot force a win (the number of stones is already a multiple
            # of 4 plus 1, e.g. 5, 9, 13), it will make a random move.
            opponent_stones_to_take = np.random.randint(1, 4)

        # Ensure the opponent's move is valid.
        opponent_stones_to_take = min(opponent_stones_to_take, self.stones_remaining)
        self.stones_remaining -= opponent_stones_to_take

        # Check if the opponent won.
        if self.stones_remaining == 0:
            reward = -1.0  # Opponent took the last stone(s), agent lost.
            terminated = True
        else:
            reward = 0.0  # Game continues, no reward or penalty yet.
            terminated = False

        observation = self.stones_remaining
        return observation, reward, terminated, False, {}

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

