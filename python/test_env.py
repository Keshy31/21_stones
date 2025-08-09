from game_env import StoneGameEnv

def test_environment():
    """
    A simple function to test the StoneGameEnv.
    """
    print("--- Testing the 21 Stone Game Environment ---")

    # 1. Create an instance of the environment.
    #    This is like setting up the game board.
    env = StoneGameEnv()

    # 2. Reset the environment to start a new game.
    #    This sets the stones to 21 and gives us the initial observation.
    print("\nStarting a new game...")
    observation, info = env.reset()
    print(f"Initial observation (stones remaining): {observation}")

    # 3. Run a few steps with random actions.
    #    This simulates a short game where the agent makes random moves.
    #    We'll loop 5 times or until the game ends.
    for i in range(5):
        print(f"\n--- Step {i + 1} ---")

        # Get a random action from the action space (0, 1, or 2)
        action = env.action_space.sample()
        print(f"Agent takes a random action: {action + 1} stone(s)")

        # Execute the action using the step function
        observation, reward, terminated, truncated, info = env.step(action)

        # 4. Print the results of the step.
        print(f"  - Opponent's move: {info.get('opponent_move', 'N/A')} stone(s) ({info.get('move_type', 'N/A')})")
        print(f"  - New observation (stones remaining): {observation}")
        print(f"  - Reward received: {reward}")
        print(f"  - Game terminated: {terminated}")

        # If the game is over, stop the test loop.
        if terminated:
            print("\nGame has ended.")
            break

    # Close the environment (optional, but good practice)
    env.close()
    print("\n--- Test complete ---")

if __name__ == "__main__":
    test_environment()


