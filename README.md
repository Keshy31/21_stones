
# 21 Stones Reinforcement Learning Project

This project demonstrates a simple yet powerful application of reinforcement learning (RL) by teaching an AI to play the "21 Stones" game. The project is divided into a Python-based simulation for training and a Rust-based implementation for an Arduino hardware version of the game.

## The Game: 21 Stones

"21 Stones" is a variant of the game of Nim. The rules are simple:
- The game starts with 21 stones.
- Two players take turns removing 1, 2, or 3 stones.
- The player who takes the last stone wins.

While the rules are simple, there is a winning strategy. This project uses reinforcement learning to allow an AI agent to "discover" this strategy through simulated gameplay.

## Project Structure

The project is organized into the following directories:

- `python/`: Contains the Python-based simulation and game engine. This is where the AI agent is trained.
- `arduino_rust/`: (Future Work) Will contain the Rust-based implementation for the Arduino hardware.
- `docs/`: Contains detailed project documentation, including a project overview.
- `diagrams/`: Contains diagrams illustrating the system architecture and hardware setup.
- `pygame_assets/`: Contains assets for the Pygame-based game engine.

## Getting Started (Python Simulation)

The core of this project is the Python simulation, which uses `pygame` for a graphical interface and a simple Q-learning algorithm to train the AI.

### Prerequisites

- Python 3.8+
- `pip` for installing packages

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd 21_stones
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    cd python
    python -m venv 21stones
    source 21stones/bin/activate  # On Windows, use `21stones\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Training the AI

To train the AI agent, run the `train.py` script:
```bash
python train.py
```
This will run thousands of simulated games, and the agent will learn the optimal strategy. The trained model, a Q-table, will be saved in the `python/runs` directory.

### Playing the Game

Once the AI is trained, you can play against it using the `game_engine.py` script:
```bash
python game_engine.py
```
This will launch a `pygame` window where you can play "21 Stones" against the trained AI.

## Hardware Implementation (Future Work)

The `arduino_rust/` directory is reserved for a future implementation of the game on Arduino hardware. The plan is to use Rust for its safety and performance benefits in embedded systems. The trained Q-table from the Python simulation will be loaded onto the Arduino to allow the AI to play in the physical world.

## Diagrams

The `diagrams/` directory contains visual aids to help understand the project:
- System architecture diagrams
- Hardware wiring diagrams (for the future Arduino implementation)

## Community and Contributions

This project is intended as an educational tool for learning about reinforcement learning. Contributions and suggestions are welcome. Please feel free to open an issue or submit a pull request.
