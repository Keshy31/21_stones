# Project Overview: Reinforcement Learning on Arduino for the 21 Stone Game

## Abstract
This project demonstrates a simple reinforcement learning (RL) application on an Arduino microcontroller, using a physical "21 stone game" (an enhanced subtraction variant of Nim) to showcase how machines can "learn" from experience. The game starts with 21 "stones" represented by LEDs, where players alternate taking 1, 2, or 3 stones via buttons, and the one who takes the last stone wins. The AI agent, powered by a Q-learning algorithm, begins with random moves but improves over simulated games to counter human strategies. Training occurs in a Python simulation accelerated by PufferLib for efficiency, with the learned policy deployed to the Arduino for real-time play. A live USB serial connection between the Arduino and laptop enables seamless loading of the Q-table (weights) and real-time logging of gameplay, wins/losses, and moves—ideal for demo videos and educational logging. This setup keeps things accessible for laypeople: They see the AI adapt and win against predictable play after a few simulated rounds, with PC logs and visuals illustrating the process. A Python-based game engine using Pygame provides a software-only playable version for on-screen play, testing, and hybrid demos, enhancing visual appeal for videos and social media. The Arduino code is implemented in Rust for safer memory handling, modern features, and enhanced reliability on resource-constrained hardware, leveraging the officially maintained AVR support in the embedded Rust ecosystem as of 2025. The project is phased with a complete software/simulation solution first (including the game engine), followed by hardware for demonstration, with future extensions for live training.

## Introduction
The 21 Stone Game RL Project bridges software simulation and hardware execution to make reinforcement learning accessible. Inspired by classic RL tutorials (e.g., enhanced Nim variants), it recreates a turn-based game on Arduino hardware while adding a novel twist: Using PufferLib (latest version 3.0 as of 2025, with core algorithmic improvements) for parallel simulation training against varied opponent strategies, resulting in a more robust AI after a few games. A live serial connection enhances demos by allowing seamless Q-table loading and real-time gameplay logging on the laptop.

The core goal is educational: Demonstrate to non-experts how an AI "learns" by playing virtual games on a computer, then applies that knowledge on a physical device to beat humans at a simple yet strategic game. For example, if a human always takes 2 stones, the AI learns to exploit this after a few simulated rounds, turning losses into wins. Live logging and the Pygame engine show moves, scores, and outcomes in real-time on the PC, making it perfect for demo videos (e.g., screen recordings of animated gameplay + log window) and social media posts (e.g., GIFs of AI adaptations).

Key features:
- **Hardware Simplicity:** 21 LEDs, 3 buttons, Arduino Uno—easy to build with ~$25 in parts.
- **Software Efficiency:** Python sim with PufferLib for quick training (minutes on a laptop).
- **Game Engine:** Python with Pygame (version 2.6.0 as of 2025) for a visual, playable software version—standalone or hybrid with hardware.
- **Live Connection:** USB serial for loading Q-table and logging gameplay/wins/losses.
- **Language Flexibility:** Arduino code in Rust, utilizing crates like `arduino-hal` for safe and efficient embedded programming. This choice aligns with Rust's growing adoption in embedded systems for 2025, including official AVR maintenance.
- **Novelty for Community:** Combines tabular Q-learning with vectorized training, live serial integration, and a Pygame engine; shareable as a GitHub repo with code, diagrams, and a beginner's guide.
- **Scalability:** Extendable to more complex games (e.g., Rock-Paper-Scissors variant) or robotics later.

This document serves as a white paper outline, suitable for inclusion in a Git repository alongside the README.md.

## Glossary
To keep explanations accessible, here's a simple breakdown of key terms (avoiding heavy jargon):
- **Reinforcement Learning (RL):** A type of machine learning where an "agent" (like our AI) learns by trying actions, getting rewards (wins) or penalties (losses), and improving over time—like training a pet with treats.
- **Q-Learning:** A basic RL method using a "score table" (Q-table) to rate how good each move is in each game situation. The AI updates scores based on outcomes, picking higher-scored moves more often.
- **Agent:** The AI player in the game, which makes decisions.
- **State:** The current game situation, e.g., number of stones left (0-21).
- **Action:** What the AI does, e.g., take 1, 2, or 3 stones.
- **Reward:** Feedback: +1 for winning, -1 for losing, 0 otherwise.
- **Policy:** The AI's strategy, derived from the Q-table (e.g., "In state 3, always take 3").
- **Epsilon-Greedy:** A way to balance trying new things (random actions with probability epsilon) and using what works (best Q-score).
- **PufferLib:** A Python library that speeds up training by running many simulated games at once (parallel), like practicing 32 matches simultaneously. Updated to version 3.0 in 2025 with improvements for efficiency.
- **Gymnasium (Gym):** A Python toolkit for creating simulated environments, like a virtual playground for the game.
- **Episode:** One full game, from 21 stones to 0.
- **Convergence:** When the Q-table stabilizes, meaning the AI has "learned" and stops changing much.
- **Serial Connection:** A simple USB link between Arduino and laptop for sending data (e.g., Q-table) and logging (e.g., real-time game updates).
- **Rust:** The programming language for the Arduino code, focused on safety and performance; requires extra setup but prevents common bugs like memory errors. Supports AVR via official tools in 2025.
- **Pygame:** A Python library for creating 2D games and visuals, used here for the software game engine to render stones, buttons, and logs on-screen.

No other acronyms are used heavily; concepts like "vectorized environments" mean running multiple sims in parallel for speed.

## System Overview and Interconnections
The project consists of two interconnected systems: a Python-based simulation and game engine for training/play, and an Arduino-based hardware for real-world demonstration. They connect live via USB serial communication, enabling seamless Q-table loading from PC to Arduino and real-time logging of gameplay data back to the PC. The Pygame engine adds a visual layer for software-only or hybrid modes.

### High-Level Components
1. **Simulation and Game Engine System (Python on PC):**
   - **Purpose:** Train the AI in a virtual environment to build the Q-table, and provide a playable software version for demos.
   - **Key Parts:**
     - Gymnasium Env: Simulates the 21 stone game, including opponent moves (varied strategies like 25-75% optimal).
     - PufferLib: Wraps the env for parallel training (e.g., 32 games at once), using CleanRL (updated implementations as of 2025) for Q-learning updates.
     - Trainer: Runs episodes, updates Q-table based on rewards, and exports the final table.
     - Pygame Engine: Renders the game visually (e.g., row of stone icons, clickable/touchable buttons for actions, real-time logs and Q-value displays). Supports modes: Human vs AI, AI vs AI, or hybrid with hardware.
     - Serial Loader/Logger: A Python script that sends the Q-table to Arduino over serial, receives live game data (moves, states, wins/losses), and integrates with the Pygame engine for display/logging.
   - **Output:** A 22x3 Q-table (states 0-21, actions take1/take2/take3) as printed numbers, plus live visuals/logs during play.

2. **Hardware System (Arduino):**
   - **Purpose:** Run the trained AI in physical play against a human, with live data exchange and optional sync to Pygame engine.
   - **Key Parts:**
     - LEDs (21): Visualize remaining stones (lit = stone present).
     - Buttons (3): Human inputs for take 1, 2, or 3.
     - Arduino Uno: Processes inputs, uses Q-table to choose AI moves, updates LEDs, and communicates via serial (e.g., receives Q-table, sends logs to Pygame).
   - **Input/Output:** Human presses buttons; Arduino responds by turning off LEDs and checking win/loss; serial sends/receives data, with Pygame mirroring the state visually.

### Interconnections
- **Live Connection:** USB cable (standard Arduino upload cable) enables bidirectional serial communication at 9600 baud. Python uses `pyserial` library to connect (e.g., `ser = serial.Serial('COM3', 9600)`).
  - **Seamless Loading:** After sim training, Python script sends Q-table values over serial (e.g., as comma-separated strings). Arduino receives and loads into its array without manual copy-paste.
  - **Live Logging and Sync:** During play, Arduino sends real-time data (e.g., "Human took 1, State:20, AI took 3, State:17") to PC. Pygame engine displays this in a GUI window with animations, stones visuals, wins/losses tallied, and optional Q-heatmaps. For demos, log to file/video capture.
  - **Protocol:** Simple text commands, e.g., Arduino sends "LOG:Move,Human,1" or "WIN:AI"; PC sends "LOAD:Q,0,0.5,1.0,...".
- **Data Flow:** Simulation/Engine → Hardware (Q-table load) ↔ Logging/Sync (bidirectional during play).
- **Training Loop (Sim/Engine Only):** Env provides state → Agent picks action (epsilon-greedy) → Env simulates opponent and reward → Update Q-table. Repeat for 1000-2000 episodes; visualize in Pygame.
- **Inference Loop (Hardware with Engine Sync):** Read button (human move) → Update state/LEDs → Send log to PC → Pygame updates visuals → If AI turn, query Q-table for best action → Execute (turn off LEDs) → Send log → Check win/loss → Send result → Pygame shows outcome.
- **Robustness Link:** Sim uses varied opponents to make Q-table general; live logging and Pygame verify real human play.
- **Debug Link:** Serial handles both loading and logs; Pygame shows everything in one window for demos.
- **No Hardware Training:** Arduino doesn't learn online (limited power); sim handles that, with serial for deployment/logging. Pygame enables software-only training demos.

This live connection and Pygame engine make demos engaging: Play on-screen or hardware while the GUI shows narrated logs (e.g., "AI chose take 3 because Q-score=0.8"), perfect for videos.

## Hardware Setup
- **Components:**
  - Arduino Uno (or compatible).
  - 21 LEDs (e.g., red) + 21 resistors (220Ω).
  - 3 Push Buttons + 3 pull-down resistors (10kΩ).
  - Breadboard and jumper wires.
  - USB Cable: For power, code upload, and live serial connection.
- **Wiring Diagram:** (Describe or include ASCII/link to image in Git)
  - LEDs: Anodes to pins 2-22 (using digital pins; note Uno has up to pin 13 standard, extend with shift registers if needed for simplicity).
  - Button 1 (take 1): Pin A0 to button to GND; pull-down from A0 to GND (analog as digital).
  - Button 2 (take 2): Pin A1 similar.
  - Button 3 (take 3): Pin A2 similar.
- **Serial Setup:** No extra hardware—use built-in USB serial (initialize with equivalent of `Serial.begin(9600)` in Rust via hardware abstraction crates).
- **Build Time:** 45-75 minutes.
- **Cost:** ~$10-15 (assuming Arduino owned).
- **Power:** USB from PC for demo (enables live connection).

## Software Simulation with PufferLib and Pygame Engine
- **Environment:** Python 3.x; install `gymnasium`, `pufferlib`, `cleanrl`, `pyserial`, `pygame` (version 2.6.0) via pip.
- **Custom Env:** StoneGameEnv: Simulates game, opponent as probabilistic "perfect" player.
- **Training:** Use PufferLib to vectorize (parallel envs), CleanRL for Q-learning. Parameters: alpha=0.5 (learning rate), gamma=0.9 (future discount), epsilon=0.3 decaying to 0.05.
- **Pygame Integration:** Extend sim into `game_engine.py`: Render stones as graphical elements (e.g., circles or images), add interactive buttons, display state/logs/Q-table heatmap (using Matplotlib embedded). Modes include standalone play, training visualization, and serial sync.
- **Serial Integration:** Post-training, the engine loads Q-table via serial and starts listening for logs, updating visuals in real-time.
- **Process:** Run 1000-2000 episodes; AI plays against sim opponents. Q-table converges (e.g., high scores for winning moves like take 3 from 3).
- **Output:** Printed Q-table for verification; live visuals/logs during play (e.g., GUI: "Game 1: Human win, Score: AI 3/10").
- **Time:** Minutes on laptop, thanks to parallel sims.
- **Code:** Extend previous snippets; use Pygame for 2D rendering to create appealing demos.

## Arduino Implementation
- **Code Base:** Rust, using embedded crates for safer code and modern features. Leverage the official AVR support in Rust as of 2025 for reliable compilation and execution on the ATmega328P (Arduino Uno's microcontroller). Key crates include `arduino-hal` for hardware abstractions (e.g., pins, serial), `avr-device` for peripherals, and `ufmt` for lightweight formatting in no_std environments.
  - **Setup:** Install Rust toolchain via `rustup`, add AVR target with `rustup target add avr-unknown-gnu-atmega328`, use `cargo` to build. Flash with `cargo embed` (via `probe-rs`) or `avrdude`. This provides safer alternatives to C++ equivalents, with compile-time checks for memory safety in Q-table arrays and button debouncing.
  - **Trade-offs:** More modern and safe than C++, but requires Rust knowledge and extra tools; performance is similar, with potential for better optimizations via Rust's ecosystem.
- **Logic:** Human turn: Wait for button input (using safe polling or interrupts), update stones/LEDs. AI turn: Choose action via max Q (safely indexed array). Send logs via serial write (e.g., equivalent to `writeln!(uart, "LOG:State,21,Human,1")`).
- **Serial Handling:** On setup, listen for "LOAD:" commands to receive Q-table (parse strings safely). During loop, send updates using Rust's no-panic guarantees.
- **Deployment:** Build and flash via `cargo embed`; run Pygame engine to load Q-table and log/sync.
- **Extensions:** Log wins/losses tally (e.g., "WINS:AI,5,HUMAN,3"). For advanced debugging, integrate `defmt` for structured logging if needed.

## Training and Deployment Workflow (Phased)
1. **Phase 1: Software & Simulation (Complete Solution with Pygame Engine)**
   - Install Python deps including Pygame.
   - Build custom Gym env for 21 game.
   - Train with PufferLib/CleanRL (vary opponents); visualize progress in Pygame (e.g., animated episodes, win rate plots).
   - Develop Pygame engine: Standalone playable game (human vs AI via mouse/keyboard), load/export Q-table, integrate training visuals.
   - Test: Play sim games in GUI; export Q-table.
   - No hardware—demo via Pygame window for videos/social (e.g., GIFs of AI learning).

2. **Phase 2: Hardware Demo Integration**
   - Setup Rust toolchain, add AVR target, and required crates (e.g., via `cargo add arduino-hal`).
   - Build & Flash: Use `cargo build --release` and flash to Arduino.
   - Connect & Load: Plug USB, run Pygame engine—it sends Q-table seamlessly over serial.
   - Play & Log/Sync: Start game on hardware; Pygame displays live moves, states, visuals (mirroring LEDs), and win/loss in GUI. Logs save to file for videos.
   - Iterate: Retrain sim if needed, reload via serial without re-flashing code.

3. **Future: Live Training for Videos**
   - After hardware/software games, feed human moves back to sim as new opponent data via Pygame interface.
   - Retrain live (minutes), reload Q-table—show adaptation in videos (e.g., "AI improves mid-demo").
   - Optional: Slow online updates on Arduino (e.g., after each game); hybrid with PufferLib for speed.

## Demo and Educational Value
- **For Laypeople:** "The AI learns like a kid playing games: Tries random moves in sim, remembers what wins, then beats you—watch it adapt live in the app or on hardware!"
- **Show Learning:** Pygame visuals for Q-table evolution and animated games; on hardware, play 5-10 rounds to show wins increase against fixed strategies, with GUI narrating (e.g., "AI won! Total: AI 4, Human 1").
- **Demo Videos:** Record Pygame gameplay + hardware cam (e.g., via OBS); visuals make it easy to explain "See the stones vanish as the AI chooses based on learned scores?" Ideal for social media: Short clips/GIFs of adaptations.
- **White Paper Tie-In:** This overview expands to discuss RL basics, with visuals (e.g., Q-table heatmaps, Pygame screenshots). Highlight Rust's role in safe embedded AI and Pygame's appeal for demos.

## Community Contribution
- **Git Repo Structure:** README.md (quick start), whitepaper.md (this doc), code folders (python/, arduino_rust/), diagrams, pygame_assets/.
- **Novelty:** PufferLib integration for fast, robust training with Pygame visuals and live serial logging in embedded RL—share on GitHub, Arduino forums, or RL subreddits as a beginner tutorial. Emphasize Rust's production-ready status in embedded systems for 2025.
- **Future:** Extend to multiplayer or robotics, leveraging Rust's concurrency and Pygame for prototyping.

This project embodies "simple yet insightful" ML on hardware—ready for your Git!