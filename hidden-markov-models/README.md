# Hidden Markov Models - Grid World Localization

## Assignment Overview

This project implements Hidden Markov Models (HMMs) for robot localization in a grid world environment. The assignment required implementing probabilistic inference algorithms (forward, backward, filtering, smoothing) and parameter learning using the Baum-Welch algorithm.

## What This Demonstrates

This project demonstrates:

- **Hidden Markov Models**: Implementation of HMMs for state estimation in partially observable environments
- **Probabilistic Inference Algorithms**:
  - Forward algorithm for computing forward probabilities
  - Backward algorithm for computing backward probabilities
  - Filtering for estimating current state given observations up to current time
  - Smoothing for estimating past states given all observations
- **Parameter Learning**: Baum-Welch algorithm for learning observation probabilities from data
- **Grid World Navigation**: Application of HMMs to robot localization with noisy sensor observations
- **Visualization**: Animated visualization of belief states over time

## What Was Provided

The following components were provided at the start of the assignment:

- **`utils.py`**: Utility functions including:
  - `loc_error()`: Calculates localization error between belief states and true trajectory
  - `inference()`: Runs filtering and smoothing inference experiments across multiple epsilon values
  - `learning()`: Generates observations and runs Baum-Welch learning
  - `visualize_one_run()`: Creates animated visualization of belief states over time
  
- **`run_experiments.py`**: Command-line interface for running experiments with various modes (originally provided as `main.py`)

- **`gridworld_hmm.py`**: Class structure with method signatures and docstrings for:
  - `Gridworld_HMM` class initialization
  - `neighbors()` method for finding valid neighboring cells
  - Method signatures for transition/observation matrix initialization
  - Method signatures for forward, backward, filtering, smoothing, and Baum-Welch algorithms

- **Function signatures and docstrings**: The structure and expected behavior of functions to implement

## What I Contributed

I implemented the core HMM algorithms in `gridworld_hmm.py`:

### 1. **Transition Matrix Initialization** (`initT`)
   - Creates an n×n transition matrix where n is the grid size
   - Computes transition probabilities based on valid neighboring cells
   - Each cell transitions uniformly to its valid neighbors (including staying in place)
   - Handles wall constraints (walls are impassable)

### 2. **Observation Matrix Initialization** (`initO`)
   - Creates a 16×n observation probability matrix
   - Models sensor observations as 4-bit binary strings representing presence/absence of neighbors in each direction
   - Accounts for sensor noise using epsilon parameter
   - Computes observation probabilities using binomial distribution based on Hamming distance between true and observed bitstrings

### 3. **Forward Algorithm** (`forward`)
   - Performs one iteration of the forward algorithm
   - Computes forward probabilities: α_t = P(observation_t | state_t) × Σ(α_{t-1} × P(state_t | state_{t-1}))
   - Updates belief state given new observation

### 4. **Backward Algorithm** (`backward`)
   - Performs one iteration of the backward algorithm
   - Computes backward probabilities: β_t = Σ(β_{t+1} × P(observation_{t+1} | state_{t+1}) × P(state_{t+1} | state_t))
   - Used in conjunction with forward algorithm for smoothing

### 5. **Filtering** (`filtering`)
   - Performs filtering over all observations
   - Estimates current state distribution given observations up to current time
   - Uses forward algorithm iteratively across all time steps
   - Returns alpha vectors and belief states at each timestep

### 6. **Smoothing** (`smoothing`)
   - Placeholder implementation for smoothing algorithm
   - Would combine forward and backward probabilities to estimate past states given all observations

### 7. **Baum-Welch Algorithm** (`baum_welch`)
   - Placeholder implementation for parameter learning
   - Would learn observation probabilities from observation sequences using expectation-maximization

## How to Run

### Prerequisites

```bash
pip install numpy matplotlib
```

### Basic Usage

```bash
python run_experiments.py [options]
```

### Arguments

#### `-m` (Mode)
Specifies the experiment mode:

- **`-m 0`** or omitted (default): Run filtering and smoothing inference experiments
  - Tests multiple epsilon values (0.4, 0.2, 0.1, 0.05, 0.02, 0)
  - Plots filtering and smoothing localization error over time
  
- **`-m 1`**: Animate one episode
  - Visualizes belief states over time for a single trajectory
  - Shows how belief distribution evolves as observations are received
  
- **`-m 2`**: Learn observation probabilities
  - Runs Baum-Welch algorithm to learn observation probabilities
  - Displays learned observation matrix and log likelihood convergence

#### `-t` (Time Steps)
Number of time steps for the experiment (default: 50)

- Example: `-t 100` (runs for 100 time steps)

#### `-n` (Number of Episodes)
Number of episodes for inference experiments (default: 500, only used with `-m 0`)

- Example: `-n 1000` (averages results over 1000 episodes)

#### `-e` (Epsilon)
Epsilon parameter for sensor noise (default: 0.0)

- Controls probability of sensor error in observations
- Used with `-m 1` (animation) or `-m 2` (learning)
- Example: `-e 0.1` (10% sensor error probability)

### Example Commands

```bash
# Run filtering and smoothing experiments (default mode)
python run_experiments.py

# Animate one episode with epsilon=0.1 for 100 time steps
python run_experiments.py -m 1 -e 0.1 -t 100

# Learn observation probabilities with epsilon=0.2
python run_experiments.py -m 2 -e 0.2

# Run inference with 1000 episodes and 75 time steps
python run_experiments.py -m 0 -n 1000 -t 75
```

## Output

The program outputs different results depending on the mode:

### Mode 0 (Inference Experiments)
- Two plots showing filtering and smoothing localization error over time
- Each plot contains multiple curves for different epsilon values
- Error is measured as L1 distance between belief distribution and true state

### Mode 1 (Animation)
- Animated visualization showing:
  - Current belief state distribution (heatmap)
  - True robot position (red dot)
  - Evolution over time

### Mode 2 (Learning)
- Two plots:
  - Learned observation probability matrix (heatmap)
  - Baum-Welch log likelihood convergence over iterations

## Grid World Environment

The grid world is a 4×16 grid with:
- **Walls**: Impassable cells that block movement and cannot be occupied
- **Free cells**: Valid positions where the robot can move
- **Movement**: Robot can move to adjacent cells (including staying in place)
- **Observations**: 4-bit binary strings indicating presence/absence of neighbors in each cardinal direction
- **Sensor noise**: Controlled by epsilon parameter (probability of bit flip in observation)

## Project Structure

```
hidden-markov-models/
├── README.md                 # This file
├── run_experiments.py        # Command-line interface
├── gridworld_hmm.py          # Core HMM algorithm implementations
└── utils.py                  # Provided utility functions
```

## Algorithm Details

### Hidden Markov Models
An HMM consists of:
- **States**: Grid cell positions (n total states)
- **Observations**: 4-bit sensor readings (16 possible observations)
- **Transition probabilities**: P(state_{t+1} | state_t) - how the robot moves
- **Observation probabilities**: P(observation_t | state_t) - what sensor readings are expected in each state

### Forward Algorithm
Computes P(state_t | observations_{1:t}) using:
```
α_t(state) = P(obs_t | state) × Σ_{prev_state} α_{t-1}(prev_state) × P(state | prev_state)
```

### Backward Algorithm
Computes backward probabilities for smoothing:
```
β_t(state) = Σ_{next_state} β_{t+1}(next_state) × P(obs_{t+1} | next_state) × P(next_state | state)
```

### Filtering
Estimates current state given all observations up to current time:
- Uses forward algorithm iteratively
- Normalizes probabilities at each step

### Smoothing
Estimates past states given all observations:
- Combines forward and backward probabilities
- More accurate than filtering for past states

### Baum-Welch Algorithm
Expectation-Maximization algorithm for learning HMM parameters:
- E-step: Compute expected state sequences using forward-backward algorithm
- M-step: Update observation probabilities based on expected counts
- Iterates until convergence

## Notes

- The grid world configuration (walls and shape) is hardcoded in `run_experiments.py`
- Sensor observations are modeled as noisy 4-bit binary strings
- Epsilon parameter controls sensor noise: higher epsilon = more noise
- Filtering error typically decreases over time as more observations are received
- Visualization requires matplotlib and may take time for long sequences

## Academic Integrity Notice

**This code is shared for portfolio purposes only.** 

This repository contains completed homework assignments from my coursework. While I'm sharing this work to demonstrate my programming and algorithm implementation skills, I want to emphasize:

- **Do not copy this code for your own assignments** - this violates academic integrity policies
- **Use this as inspiration, not a solution** - understand the concepts and implement your own solutions
- **Respect your institution's honor code** - academic dishonesty has serious consequences

If you're an instructor and believe this code should not be public, please contact me and I will remove it.

---

**Language**: Python 3  
**Libraries**: NumPy, Matplotlib
