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
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="21_stones",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="StoneGame",
        help="the id of the environment")
    parser.add_argument("--total-episodes", type=int, default=10000,
        help="total episodes of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.5,
        help="the learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.9,
        help="the discount factor gamma")
    parser.add_argument("--start-e", type=float, default=1.0,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.8,
        help="the fraction of `total-episodes` it takes to decay epsilon to `end-e`")

    args = parser.parse_args()
    # fmt: on
    return args

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(end_e, start_e + slope * t)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    # env setup
    env = StoneGameEnv()

    # Q-table setup
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    start_time = time.time()

    for episode in range(args.total_episodes):
        obs, info = env.reset(seed=args.seed)
        terminated = False
        total_reward = 0

        while not terminated:
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_episodes, episode)
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[obs, :])
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Q-learning update
            old_value = q_table[obs, action]
            next_max = np.max(q_table[next_obs, :])
            
            new_value = old_value + args.learning_rate * (reward + args.gamma * next_max - old_value)
            q_table[obs, action] = new_value
            
            obs = next_obs
            total_reward += reward

        if episode % 100 == 0:
            writer.add_scalar("charts/episodic_return", total_reward, episode)
            writer.add_scalar("charts/epsilon", epsilon, episode)
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Save the Q-table
    model_dir = pathlib.Path(f"runs/{run_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    q_table_path = model_dir / "q_table.npy"
    np.save(q_table_path, q_table)
    print(f"Q-table saved to {q_table_path}")

    # Also save as a human-readable text file
    q_table_txt_path = model_dir / "q_table.txt"
    np.savetxt(q_table_txt_path, q_table, fmt="%.3f")
    print(f"Q-table saved to {q_table_txt_path}")


    env.close()
    writer.close()
