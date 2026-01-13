# Search Algorithms and Game AI - Sudoku Solver and Othello Player

## Assignment Overview

This project implements two different AI approaches: a simulated annealing solver for Sudoku puzzles and a Monte Carlo Tree Search (MCTS) player for the game of Othello. The assignment required implementing core algorithms for both local search optimization and game tree search techniques.

## What This Demonstrates

This project demonstrates:

- **Simulated Annealing**: A probabilistic local search algorithm for solving Sudoku puzzles, showing how optimization problems can be solved using temperature-based acceptance criteria
- **Monte Carlo Tree Search (MCTS)**: A game-playing algorithm that balances exploration and exploitation using the UCT (Upper Confidence Bounds applied to Trees) formula
- **Local Search**: Using neighbor states (swaps within subgrids) and evaluation functions (error counting) to solve constraint satisfaction problems
- **Game Tree Search**: Tree expansion, simulation, and backpropagation for decision-making in adversarial games
- **Algorithm Analysis**: Performance tuning through parameter experimentation (temperature schedules, rollout counts, UCT exploration weight)

## What Was Provided

The following components were provided at the start of the assignment:

### Sudoku Solver (`sudoku_simulated_annealing_solver.py`)
- **`generate()`**: Generates random Sudoku puzzles with a specified number of clues (credit: StackOverflow)
- **`initialize()`**: Fills empty cells in each major subgrid to create valid initial states
- **`successors()`**: Generates successor states by swapping two non-clue entries within a major subgrid
- **`num_errors()`**: Computes the total number of errors (missing values) across all rows and columns
- **Main function structure**: Command-line interface with argument parsing and visualization support

### Othello Game Infrastructure
- **`othello_game.py`**: Game manager maintaining board state, score, and player interfaces (credit: original instructor)
- **`othello_shared.py`**: Shared utility functions including:
  - `find_lines()`: Finds lines of pieces that would be captured
  - `get_possible_moves()`: Returns all legal moves for a player
  - `play_move()`: Executes a move and flips captured pieces
  - `compute_utility()`: Calculates game score (dark - light pieces)
- **`othello_gui.py`**: Graphical user interface for playing Othello (credit: original instructor)
- **`randy_ai.py`**: Random-move AI player for testing and comparison (credit: original instructor)
- **`mcts_tests.py`**: Unit tests for MCTS functions

### MCTS Player Infrastructure (`monte_carlo_player.py`)
- **`Node` class**: Tree node structure with state, player, parent, children, value, and visit count (N)
- **`get_child()` method**: Retrieves child node by state
- **`mcts()` function**: Main MCTS loop structure (executes select, expand, simulate, backprop repeatedly)
- **`run_ai()` function**: Communication interface with game manager

## What I Contributed

I implemented the core algorithms in the following files:

### 1. **Simulated Annealing Solver** (`simulated_annealing()` in `sudoku_simulated_annealing_solver.py`)
   - Implements the simulated annealing algorithm with temperature schedule: `T = startT * (decay^iter)`
   - Uses probabilistic acceptance criteria: accepts better states always, accepts worse states with probability `exp(ΔE/T)`
   - Tracks error history throughout the search
   - Terminates when temperature drops below tolerance, no successors available, or solution found (zero errors)

### 2. **Monte Carlo Tree Search Functions** (`monte_carlo_player.py`)
   - **`select()`**: Traverses the tree using UCT (Upper Confidence Bounds for Trees) formula:
     - `UCT = value + α * sqrt(ln(parent.N) / child.N)`
     - Returns leaf node or node with unexpanded children
   - **`expand()`**: Adds a new child node to the tree for an unexplored successor state
   - **`simulate()`**: Runs a random playout from the leaf node to terminal state, returns utility
   - **`backprop()`**: Backpropagates utility values up the tree, updating node values and visit counts
     - Handles player perspective: negates utility for dark player (minimizer), uses direct utility for light player (maximizer)
     - Updates average value: `value = (value * (N-1) + player_utility) / N`

## How to Run

### Prerequisites

```bash
pip install numpy matplotlib
```

For Othello GUI, you may need:
```bash
pip install tkinter  # or install via system package manager
```

### Sudoku Solver

Run the Sudoku solver with:

```bash
python sudoku_simulated_annealing_solver.py -n <grid_size> -c <num_clues> [options]
```

#### Arguments

- **`-n`** (required): Grid size (e.g., `9` for 9×9 Sudoku)
- **`-c`** (required): Number of clues (pre-filled cells)
- **`-s`** (optional): Starting temperature (default: 100)
- **`-d`** (optional): Decay rate for temperature (default: 0.5)
- **`-b`** (optional): Batch mode - run multiple searches and show histogram

#### Examples

```bash
# Solve a single 9×9 puzzle with 40 clues
python sudoku_simulated_annealing_solver.py -n 9 -c 40

# Solve with custom temperature parameters
python sudoku_simulated_annealing_solver.py -n 9 -c 40 -s 50 -d 0.95

# Run batch of 30 puzzles and show histogram
python sudoku_simulated_annealing_solver.py -n 9 -c 40 -b 30
```

### Othello Player

Run the Othello game with:

```bash
python othello_gui.py [options]
```

#### Options

- **`-p1`**: AI agent file for player 1 (dark)
- **`-p2`**: AI agent file for player 2 (light)
- **`-b`**: Board size (default: 4)

#### Examples

```bash
# Play against yourself (human vs human)
python othello_gui.py

# Play against random AI
python othello_gui.py -p2 randy_ai.py

# Play against MCTS AI
python othello_gui.py -p2 monte_carlo_player.py

# Watch MCTS vs Random AI
python othello_gui.py -p1 monte_carlo_player.py -p2 randy_ai.py

# Use larger board
python othello_gui.py -b 8 -p1 monte_carlo_player.py -p2 randy_ai.py
```

### Testing MCTS Functions

Run unit tests:

```bash
python mcts_tests.py
```

## Project Structure

```
local-search-and-game-ai/
├── README.md                          # This file
├── sudoku_simulated_annealing_solver.py  # Sudoku solver implementation
├── monte_carlo_player.py              # MCTS Othello player implementation
├── othello_game.py                    # Game manager and player interfaces
├── othello_shared.py                  # Shared game utility functions
├── othello_gui.py                     # Graphical user interface
├── randy_ai.py                        # Random-move AI player
└── mcts_tests.py                      # Unit tests for MCTS functions
```

## Algorithm Details

### Simulated Annealing for Sudoku

- **State Space**: Valid Sudoku boards where each major subgrid contains all numbers 1 to n
- **Neighbor Generation**: Swap two non-clue entries within the same major subgrid
- **Evaluation Function**: Count missing values across all rows and columns
- **Temperature Schedule**: Exponentially decaying temperature: `T = startT * (decay^iter)`
- **Acceptance Probability**: 
  - Always accept improving moves (ΔE > 0)
  - Accept worsening moves with probability `exp(ΔE/T)` when ΔE ≤ 0
- **Termination**: Temperature < tolerance, no successors, or solution found

### Monte Carlo Tree Search for Othello

- **Tree Structure**: Nodes represent game states with associated value estimates and visit counts
- **UCT Selection**: Balances exploitation (high value) and exploration (low visit count) using:
  - `UCT = value + α * sqrt(ln(parent.N) / child.N)`
- **Four-Phase Algorithm**:
  1. **Select**: Traverse tree using UCT until leaf or unexpanded node
  2. **Expand**: Add one new child node for unexplored successor
  3. **Simulate**: Random playout to terminal state, compute utility
  4. **Backprop**: Update values and visit counts along path to root
- **Player Perspective**: Dark player (1) maximizes negative utility, light player (2) maximizes positive utility
- **Move Selection**: After rollouts, choose move with highest visit count

## Notes

- The Sudoku solver works best with carefully tuned temperature schedules (startT and decay)
- MCTS performance improves with more rollouts (default: 100) but computation time increases
- Larger Othello boards (>10×10) may be slow with default MCTS parameters
- The MCTS player significantly outperforms random play but may struggle against expert human players
- Sudoku puzzles with too few clues may be unsolvable with this approach

## Academic Integrity Notice

**This code is shared for portfolio purposes only.** 

This repository contains completed homework assignments from my coursework. While I'm sharing this work to demonstrate my programming and algorithm implementation skills, I want to emphasize:

- **Do not copy this code for your own assignments** - this violates academic integrity policies
- **Use this as inspiration, not a solution** - understand the concepts and implement your own solutions
- **Respect your institution's honor code** - academic dishonesty has serious consequences

If you're an instructor and believe this code should not be public, please contact me and I will remove it.

---

**Language**: Python 3  
**Libraries**: NumPy, Matplotlib, Tkinter
