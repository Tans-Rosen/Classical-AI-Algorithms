# Multi-Armed Bandit Algorithms - Online Shortest Path Problem

## Assignment Overview

This project implements multi-armed bandit algorithms for solving an online shortest path problem in a grid network. The assignment required implementing and comparing different exploration-exploitation strategies including epsilon-greedy, epsilon-decaying, and Upper Confidence Bound (UCB) algorithms.

## What This Demonstrates

This project demonstrates:

- **Multi-Armed Bandit Problem**: Application of bandit algorithms to sequential decision-making in uncertain environments
- **Exploration-Exploitation Trade-off**: Balancing between exploring unknown paths and exploiting known good paths
- **Online Learning Algorithms**:
  - **Epsilon-Greedy**: Fixed exploration probability with greedy exploitation
  - **Epsilon-Decaying**: Adaptive exploration that decreases over time
  - **Upper Confidence Bound (UCB)**: Confidence-based selection that balances exploration and exploitation
- **Regret Analysis**: Measuring cumulative regret to evaluate algorithm performance
- **Shortest Path Problem**: Using bandit algorithms to learn the optimal path in a stochastic network
- **Stochastic Environments**: Handling reward distributions with uncertainty

## What Was Provided

The following components were provided at the start of the assignment:

- **`graph.py`**: `Graph` class including:
  - `__create_network()`: Generates a binomial bridge network with random edge weights
  - `__create_arms()`: Enumerates all possible paths from source to destination
  - `shortest_path()`: Finds the optimal path using graph algorithms
  - `shortest_path_ind()`: Returns the index of the optimal arm
  - Edge weight generation with log-normal distributions
  
- **`utils.py`**: Visualization utilities including:
  - `draw_graph()`: Draws the network graph structure
  - `highlight_shortest_path()`: Highlights the optimal path in green
  - `highlight_path()`: Highlights a specific path in coral
  - `plot_graphs()`: Creates comparison visualizations of paths found
  - `plot_results()`: Plots regret and optimal path frequency over time
  
- **`bandit_experiments.py`**: Command-line interface for running experiments with various parameter configurations (originally provided as `main.py`)

- **`Bandit` class skeleton**: Structure and helper methods (`pull_arm`, `get_path_mean`) were provided

- **Function signatures and docstrings**: The structure and expected behavior of functions to implement

## What I Contributed

I implemented the core multi-armed bandit algorithms in `bandit.py`:

### 1. **Main Simulation Loop** (`simulate`)
   - Orchestrates the bandit learning process over N time steps
   - Tracks Q-values (estimated rewards) for each arm/path
   - Calculates cumulative regret by comparing chosen arm against optimal arm
   - Updates Q-values using incremental averaging: `Q_new = Q_old + (reward - Q_old) / count`
   - Selects arms based on the chosen strategy (epsilon-greedy, epsilon-decay, or UCB)

### 2. **Epsilon-Greedy Strategy** (`choose_arm_egreedy`)
   - Implements fixed epsilon-greedy exploration-exploitation
   - With probability `epsilon`, explores by selecting a random arm
   - With probability `1 - epsilon`, exploits by selecting the arm with highest Q-value
   - Simple and effective baseline strategy

### 3. **Epsilon-Decaying Strategy** (`choose_arm_edecay`)
   - Implements adaptive epsilon-greedy with decaying exploration rate
   - Epsilon decreases over time: `epsilon(t) = min(1, (value * num_arms) / (t + 1))`
   - Balances early exploration with later exploitation
   - Adapts exploration based on time step

### 4. **Upper Confidence Bound Strategy** (`choose_arm_ucb`)
   - Implements UCB algorithm for confidence-based arm selection
   - Calculates UCB value: `Q_value + value * sqrt(log(t) / count)`
   - Selects arm with highest upper confidence bound
   - Automatically balances exploration (uncertain arms) and exploitation (high Q-value arms)
   - Provides theoretical regret guarantees

## How to Run

### Prerequisites

```bash
pip install numpy matplotlib networkx
```

### Basic Usage

```bash
python bandit_experiments.py [options]
```

### Options

#### `-w` (Bridge Width)
Specifies the width of the binomial bridge network:

- **Default**: 2
- **Example**: `-w 3` (creates a 3x3 grid with more paths)

#### `-s` (Strategy)
Specifies which bandit strategy to use:

- **`-s 0`** or omitted: Epsilon-Greedy (default)
- **`-s 1`**: Epsilon-Decaying
- **`-s 2`**: Upper Confidence Bound (UCB)

#### `-pm` (Prior Mu)
Prior mean parameter for edge weight generation:

- **Default**: -0.5
- **Example**: `-pm -1.0` (lowers average edge weights)

#### `-ps` (Prior Sigma)
Prior standard deviation parameter for edge weight generation:

- **Default**: 1.0
- **Example**: `-ps 0.5` (reduces variance in edge weights)

#### `-cs` (Conditional Sigma)
Conditional standard deviation for reward generation:

- **Default**: 1.0
- **Example**: `-cs 0.5` (reduces reward variance)

#### `-m` (Simulation Runs)
Number of independent simulation runs to average:

- **Default**: 100
- **Example**: `-m 200` (more runs for better statistics)

#### `-n` (Time Steps)
Number of time steps per simulation:

- **Default**: 10000
- **Example**: `-n 5000` (shorter learning horizon)

### Example Commands

```bash
# Run epsilon-greedy with default parameters
python bandit_experiments.py

# Test UCB algorithm with 3x3 bridge
python bandit_experiments.py -s 2 -w 3

# Run epsilon-decaying with 200 simulation runs
python bandit_experiments.py -s 1 -m 200

# Test different reward variance
python bandit_experiments.py -cs 0.5 -n 5000
```

## Output

The program outputs:

- **Strategy name**: Which algorithm is being tested
- **Visualizations**:
  - Graph comparison showing optimal path vs. paths found by algorithm
  - Regret plot showing cumulative regret over time for different parameter values
  - Optimal path frequency showing how often the optimal path is selected over time

Example output:
```
========================================
Testing Epsilon-Greedy...

========================================
Plotting Results...

Done!
========================================
```

## Network Structure

The project uses a **binomial bridge** network structure:

- Grid-based directed graph from (0,0) to (width-1, width-1)
- Each edge has a stochastic weight drawn from a log-normal distribution
- Multiple paths exist from source to destination
- Each path is treated as a "arm" in the multi-armed bandit problem
- Rewards are negative path costs (lower is better, so maximizing reward minimizes cost)

## Algorithm Details

### Epsilon-Greedy
- **Exploration**: Fixed probability `epsilon` of random selection
- **Exploitation**: Select arm with highest Q-value
- **Trade-off**: Simple but may over-explore in later stages
- **Parameter**: `epsilon` (e.g., 0.1, 0.2, 0.5)

### Epsilon-Decaying
- **Exploration**: Adaptive probability that decreases over time
- **Formula**: `epsilon(t) = min(1, (c * num_arms) / t)`
- **Advantage**: More exploration early, more exploitation later
- **Parameter**: Decay constant `c` (e.g., 1, 5, 10)

### Upper Confidence Bound (UCB)
- **Selection**: `argmax(Q_value + C * sqrt(log(t) / count))`
- **Exploration**: Higher uncertainty (lower count) increases UCB value
- **Exploitation**: Higher Q-value increases UCB value
- **Theoretical**: Logarithmic regret bound
- **Parameter**: Confidence constant `C` (e.g., 1.0, 2.0, 5.0)

### Q-Value Update
All strategies use incremental averaging:
```
Q_new = Q_old + (reward - Q_old) / count
```
This provides an unbiased estimate of expected reward for each arm.

## Project Structure

```
multi-armed-bandits-and-online-learning/
├── README.md                 # This file
├── bandit_experiments.py     # Command-line interface
├── bandit.py                 # Multi-armed bandit algorithm implementations
├── graph.py                  # Network graph generation and path finding
└── utils.py                  # Visualization utilities
```

## Notes

- The optimal path is computed offline using NetworkX's shortest path algorithm
- Regret is calculated as the difference between optimal arm reward and chosen arm reward
- Multiple simulation runs are averaged to provide statistical reliability
- Edge weights follow log-normal distribution: `exp(prior_mu + prior_sigma * N(0,1))`
- Rewards are stochastic with conditional log-normal noise
- Visualization shows comparison between optimal path and algorithm-discovered paths

## Academic Integrity Notice

**This code is shared for portfolio purposes only.** 

This repository contains completed homework assignments from my coursework. While I'm sharing this work to demonstrate my programming and algorithm implementation skills, I want to emphasize:

- **Do not copy this code for your own assignments** - this violates academic integrity policies
- **Use this as inspiration, not a solution** - understand the concepts and implement your own solutions
- **Respect your institution's honor code** - academic dishonesty has serious consequences

If you're an instructor and believe this code should not be public, please contact me and I will remove it.

---

**Language**: Python 3  
**Libraries**: NumPy, Matplotlib, NetworkX
