import argparse
import os
import random
import time
from distutils.util import strtobool
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pathlib
from game_env import StoneGameEnv

def parse_args():
    # Parse command-line arguments for configuring the experiment
    # These allow customization of hyperparameters without changing the code
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment for reproducibility")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="21_stones",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    # Algorithm-specific arguments for Q-learning and the environment
    parser.add_argument("--env-id", type=str, default="StoneGame",
        help="the id of the environment")
    parser.add_argument("--total-episodes", type=int, default=10000,
        help="total number of episodes (full games) to run for training")
    parser.add_argument("--learning-rate", type=float, default=0.5,
        help="the learning rate (alpha) for updating Q-values; controls how much new information overrides old")
    parser.add_argument("--gamma", type=float, default=0.9,
        help="the discount factor gamma; determines the importance of future rewards")
    parser.add_argument("--start-epsilon", type=float, default=1.0,
        help="the starting epsilon value for exploration (starts with fully random actions)")
    parser.add_argument("--end-epsilon", type=float, default=0.05,
        help="the ending epsilon value for exploration (reduces to mostly exploiting learned policy)")
    parser.add_argument("--exploration-fraction", type=float, default=0.8,
        help="the fraction of total episodes over which epsilon decays from start to end")
    args = parser.parse_args()
    return args

def linear_schedule(start_epsilon: float, end_epsilon: float, duration: int, current_step: int):
    # Linearly decay epsilon over time to balance exploration and exploitation
    # Formula: epsilon = max(end_epsilon, start_epsilon + slope * current_step)
    slope = (end_epsilon - start_epsilon) / duration
    return max(end_epsilon, start_epsilon + slope * current_step)

if __name__ == "__main__":
    args = parse_args()
    # Create a unique name for this run to organize logs and results
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Optional: Set up Weights & Biases (wandb) for tracking if enabled
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # Set up TensorBoard writer for logging metrics like rewards and epsilon
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Seed random number generators for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create the environment: StoneGameEnv simulates the 21 stone game
    # State: number of stones left (0-21)
    # Actions: take 1, 2, or 3 stones
    # Reward: +1 for winning (taking the last stone), -1 for losing, 0 otherwise
    env = StoneGameEnv()
    
    # Initialize the Q-table: A 2D array where rows are states (0-21 stones),
    # columns are actions (0: take1, 1: take2, 2: take3). Starts with zeros.
    # Q[state, action] will store the expected future reward for that pair.
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Record start time for potential timing metrics
    start_time = time.time()
    
    # Main training loop: Run for the specified number of episodes
    for episode in range(args.total_episodes):
        # Reset the environment at the start of each episode (game starts with 21 stones)
        current_state, info = env.reset(seed=args.seed)
        done = False  # Flag for episode termination
        episode_reward = 0  # Track total reward for this episode
        
        # Loop through steps (turns) in the episode until the game ends
        while not done:
            # Calculate current epsilon using linear decay schedule
            # Epsilon controls exploration: high = more random actions, low = follow Q-table
            epsilon = linear_schedule(
                args.start_epsilon,
                args.end_epsilon,
                args.exploration_fraction * args.total_episodes,
                episode
            )
            
            # Epsilon-greedy policy: With probability epsilon, explore (random action);
            # otherwise, exploit (best action from Q-table)
            if random.random() < epsilon:
                action = env.action_space.sample()  # Random action for exploration
            else:
                action = np.argmax(q_table[current_state, :])  # Best action for exploitation
            
            # Take the action in the environment and get the results
            next_state, reward, done, truncated, info = env.step(action)
            
            # Q-learning update rule (Bellman equation):
            # New Q = Old Q + learning_rate * (reward + gamma * max_future_Q - Old Q)
            # This updates the value based on immediate reward and discounted future value
            old_q_value = q_table[current_state, action]
            max_next_q = np.max(q_table[next_state, :])  # Best Q for next state
            new_q_value = old_q_value + args.learning_rate * (reward + args.gamma * max_next_q - old_q_value)
            q_table[current_state, action] = new_q_value
            
            # Move to the next state and accumulate reward
            current_state = next_state
            episode_reward += reward
        
        # Log metrics every 100 episodes for monitoring progress
        if episode % 100 == 0:
            writer.add_scalar("charts/episodic_return", episode_reward, episode)
            writer.add_scalar("charts/epsilon", epsilon, episode)
            print(f"Episode: {episode}, Total Reward: {episode_reward}, Epsilon: {epsilon:.2f}")
    
    # Save the trained Q-table for later use (e.g., deployment to Arduino)
    model_dir = pathlib.Path(f"runs/{run_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    q_table_path = model_dir / "q_table.npy"
    np.save(q_table_path, q_table)
    print(f"Q-table saved to {q_table_path}")
    
    # Also save as a text file for easy human inspection
    q_table_txt_path = model_dir / "q_table.txt"
    np.savetxt(q_table_txt_path, q_table, fmt="%.3f")
    print(f"Q-table saved to {q_table_txt_path}")
    
    # Clean up resources
    env.close()
    writer.close()