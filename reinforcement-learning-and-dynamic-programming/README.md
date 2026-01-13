# Reinforcement Learning - Crawler Robot Control

## Assignment Overview

This project implements reinforcement learning and dynamic programming algorithms to control a crawling robot. The assignment required implementing Q-learning for online learning and value iteration for offline policy computation in a robot control task environment.

## What This Demonstrates

This project demonstrates:

- **Q-Learning**: Implementation of Q-learning, a model-free reinforcement learning algorithm, for online learning from interactions with the environment
- **Value Iteration**: Implementation of value iteration, a dynamic programming algorithm, for computing optimal policies in known MDPs
- **Epsilon-Greedy Exploration**: Epsilon-greedy action selection strategy balancing exploration and exploitation
- **Policy Extraction**: Deriving optimal policies from computed value functions
- **Reinforcement Learning Concepts**: 
  - Q-value updates using temporal difference learning
  - Discount factor (gamma) for future reward consideration
  - Learning rate (alpha) for controlling update magnitude
  - Exploration vs exploitation trade-off

## What Was Provided

The following components were provided at the start of the assignment:

- **`crawler.py`**: Robot environment and physics simulation including:
  - `CrawlingRobotEnvironment`: Manages state space discretization, state transitions, and reward computation
  - `CrawlingRobot`: Handles robot physics, position calculation, and rendering
  - State space discretization (9 arm states × 13 hand states)
  
- **`crawler_graphics.py`**: Graphical user interface including:
  - Interactive visualization of the crawling robot
  - Controls for adjusting learning parameters (alpha, epsilon, gamma)
  - Real-time display of robot position and velocity
  
- **Class structures and method signatures**: The structure and expected behavior of functions to implement in `reinforcement_learning_agent.py` and `dynamic_programming_agent.py`

## What I Contributed

I implemented the core learning algorithms in the agent files:

### 1. **Q-Learning Agent** (`reinforcement_learning_agent.py`)
   - **`choose_action()`**: Implements epsilon-greedy action selection
     - With probability epsilon: selects a random action (exploration)
     - Otherwise: selects the action with highest Q-value (exploitation)
   - **`update()`**: Implements Q-learning update rule
     - Updates Q-values using temporal difference learning: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
     - Uses greedy action selection for the next state (off-policy learning)

### 2. **Dynamic Programming Agent** (`dynamic_programming_agent.py`)
   - **`value_iteration()`**: Implements value iteration algorithm
     - Iteratively updates state values until convergence
     - Convergence detected when value changes are below threshold (10e-6)
     - Computes optimal value function: V(s) ← max_a Σ [R(s,a) + γ·V(s')]
   - **`policy_extraction()`**: Extracts optimal policy from value function
     - For each state, selects action that maximizes expected value
     - Policy: π(s) = argmax_a [R(s,a) + γ·V(s')]

## How to Run

### Prerequisites

```bash
pip install tkinter
```

(Note: `tkinter` is typically included with Python installations. If not available, install via your system package manager.)

### Basic Usage

```bash
python crawler.py [-q]
```

### Arguments

- **`-q`** (optional): Use Q-learning agent instead of dynamic programming agent
  - If omitted: Uses value iteration (dynamic programming) agent
  - If specified: Uses Q-learning agent

### Example Commands

```bash
# Run with dynamic programming agent (value iteration)
python crawler.py

# Run with Q-learning agent
python crawler.py -q
```

### Interactive Controls

The GUI provides interactive controls for adjusting learning parameters:

- **Epsilon (ε)**: Exploration rate - controls balance between exploration and exploitation
  - Increase: More random actions (more exploration)
  - Decrease: More greedy actions (more exploitation)
  - Adjust with +/- buttons

- **Gamma (γ)**: Discount factor - determines importance of future rewards
  - Increase: Values future rewards more
  - Decrease: Focuses on immediate rewards
  - Adjust with +/- buttons
  - For DP agent: Changing gamma recomputes the optimal policy

- **Alpha (α)**: Learning rate - controls step size of Q-value updates (Q-learning only)
  - Increase: Faster learning, but potentially less stable
  - Decrease: Slower but more stable learning
  - Adjust with +/- buttons

- **Speed**: Controls simulation speed
  - Increase: Faster simulation (more steps per second)
  - Decrease: Slower simulation (easier to observe)
  - Adjust with +/- buttons

## Output

The GUI displays:

- **Visualization**: Animated robot moving horizontally
- **Position**: Current x-coordinate of the robot
- **Velocity**: Instantaneous velocity (change in position per step)
- **100-step Average Velocity**: Rolling average velocity over the last 100 steps
- **Step Count**: Total number of actions taken

The robot's goal is to maximize forward movement (x-coordinate). Rewards are based on horizontal displacement.

## Project Structure

```
reinforcement-learning-and-dynamic-programming/
├── README.md                        # This file
├── crawler.py                       # Robot environment and physics
├── crawler_graphics.py              # GUI and visualization
├── reinforcement_learning_agent.py  # Q-learning agent implementation
└── dynamic_programming_agent.py     # Value iteration agent implementation
```

## Algorithm Details

### Q-Learning
- **Type**: Model-free, off-policy temporal difference learning
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
- **Exploration**: Epsilon-greedy (ε probability of random action)
- **Convergence**: Learns optimal Q-function under certain conditions (sufficient exploration, appropriate learning rate decay)

### Value Iteration
- **Type**: Dynamic programming, model-based
- **Update Rule**: V(s) ← max_a Σ [R(s,a) + γ·V(s')]
- **Convergence**: Guaranteed to converge to optimal value function
- **Policy**: Extracted after convergence by selecting greedy actions

### Robot Environment
- **State Space**: Discretized as (arm_angle_bucket, hand_angle_bucket)
  - 9 arm angle states (buckets)
  - 13 hand angle states (buckets)
  - Total: 117 states
- **Actions**: 
  - `arm-up`: Increase arm angle
  - `arm-down`: Decrease arm angle
  - `hand-up`: Increase hand angle
  - `hand-down`: Decrease hand angle
- **Reward**: Horizontal displacement (new_x - old_x)

## Notes

- The robot environment uses discrete state space for computational efficiency
- Value iteration computes the optimal policy offline (before interaction)
- Q-learning learns the optimal policy online (through interaction)
- The GUI allows real-time parameter adjustment to observe effects on learning
- For Q-learning, lower epsilon values over time typically improve performance
- The dynamic programming approach assumes perfect knowledge of the environment model

## Academic Integrity Notice

**This code is shared for portfolio purposes only.** 

This repository contains completed homework assignments from my coursework. While I'm sharing this work to demonstrate my programming and algorithm implementation skills, I want to emphasize:

- **Do not copy this code for your own assignments** - this violates academic integrity policies
- **Use this as inspiration, not a solution** - understand the concepts and implement your own solutions
- **Respect your institution's honor code** - academic dishonesty has serious consequences

If you're an instructor and believe this code should not be public, please contact me and I will remove it.

---

**Language**: Python 3  
**Libraries**: Tkinter (GUI), Math (calculations)
