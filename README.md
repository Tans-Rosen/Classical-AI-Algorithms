# Classical AI Algorithms

This repository is a curated collection of **classical artificial intelligence and machine learning algorithms**, implemented from scratch and applied to concrete problem domains. The projects span search, probabilistic reasoning, online learning, reinforcement learning, and deep learning, with a strong emphasis on **algorithmic correctness, empirical evaluation, and clean software structure**.

Each subdirectory is a self-contained project with its own README, codebase, and runnable experiments.

---

## Repository Overview

The repository covers the following core areas of classical AI:

- **Search & Planning**
- **Probabilistic Models**
- **Online Learning**
- **Reinforcement Learning & Dynamic Programming**
- **Game AI**
- **Neural Networks**

All projects are implemented in **Python 3** and emphasize:
- Clear algorithmic implementations
- Reproducible experiments
- Quantitative evaluation (regret, error, convergence, accuracy)
- Visualization where appropriate

---

## Projects

### 1. Pathfinding Algorithms — Grid World Navigation
📁 `path-finding-algorithms/`

Implements and compares classical graph search algorithms for robot navigation in grid worlds with varying terrain costs.

**Algorithms:**
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* (Manhattan & Euclidean heuristics)
- Beam Search
- Iterative Deepening A* (IDA*)

**Highlights:**
- Cost-aware pathfinding with impassable terrain
- Memory vs optimality trade-offs
- Animated visualization of node expansion
- Detailed metrics: path cost, nodes expanded, frontier size

---

### 2. Local Search & Game AI — Sudoku and Othello
📁 `local-search-and-game-ai/`

Two distinct AI paradigms applied to different problem types.

#### Sudoku Solver (Optimization)
- Simulated Annealing with temperature schedules
- Constraint satisfaction via local search
- Probabilistic acceptance of worse states

#### Othello Player (Adversarial Search)
- Monte Carlo Tree Search (MCTS)
- UCT-based selection
- Rollout simulation and backpropagation
- Competitive performance against baseline agents

---

### 3. Hidden Markov Models — Grid World Localization
📁 `hidden-markov-models/`

Implements **Hidden Markov Models (HMMs)** for robot localization under sensor noise.

**Algorithms:**
- Forward algorithm
- Backward algorithm
- Filtering
- Smoothing
- (Scaffolded) Baum-Welch parameter learning

**Features:**
- Noisy 4-bit sensor observations
- Probabilistic belief tracking
- Localization error analysis
- Animated belief-state visualization

---

### 4. Multi-Armed Bandits — Online Shortest Path Learning
📁 `multi-armed-bandits-and-online-learning/`

Applies online learning algorithms to a stochastic shortest-path problem.

**Algorithms:**
- Epsilon-Greedy
- Epsilon-Decaying
- Upper Confidence Bound (UCB)

**Key Concepts:**
- Exploration vs exploitation trade-offs
- Regret minimization
- Confidence-based decision making
- Learning optimal paths in uncertain environments

Includes extensive visualization of regret curves and path selection frequency.

---

### 5. Reinforcement Learning & Dynamic Programming — Crawler Robot
📁 `reinforcement-learning-and-dynamic-programming/`

Controls a simulated crawling robot using both model-free and model-based approaches.

**Algorithms:**
- Q-Learning (online, model-free)
- Value Iteration (offline, model-based)

**Features:**
- Discretized state space
- Epsilon-greedy exploration
- Interactive GUI with real-time parameter tuning
- Direct comparison between learning paradigms

---

### 6. Convolutional Neural Networks — Imagenette Classifier
📁 `convolutional-neural-networks/`

Implements a CNN from scratch in PyTorch for image classification on the Imagenette dataset.

**Highlights:**
- Modular PyTorch codebase (data, model, training, evaluation)
- Train/validation/test pipeline
- Checkpointing (best vs last)
- CLI-driven experimentation
- Clean refactor from notebook to production-style code

---

## Technologies & Tools

- **Language:** Python 3
- **Libraries:** NumPy, Matplotlib, NetworkX, PyTorch, torchvision, Tkinter
- **Paradigms:** Search, Probabilistic Inference, Online Learning, Reinforcement Learning, Deep Learning
- **Focus:** Algorithmic clarity, evaluation rigor, and reproducibility

---

## Design Philosophy

This repository emphasizes:
- Implementing algorithms **from first principles**
- Separating core logic from experimentation code
- Making algorithmic trade-offs explicit
- Treating academic work as production-quality portfolio material

Each project README includes:
- Algorithm explanations
- Experimental setup
- Usage instructions
- Output interpretation

---

## Academic Integrity Notice

**This repository is shared strictly for portfolio and educational purposes.**

These projects originated as coursework assignments and have been refactored into standalone codebases.  
Do **not** submit this code or derivatives for academic credit.

If you are an instructor and believe this repository should not be public, please contact me and I will remove it.

---

## Author

**Tans Rosen**

This repository reflects a deep interest in classical AI foundations, algorithmic problem-solving, and building systems that connect theory with practice.
