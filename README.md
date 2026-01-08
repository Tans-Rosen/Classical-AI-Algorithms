# Pathfinding Algorithms - Grid World Navigation

## Assignment Overview

This project implements various pathfinding algorithms for robot navigation in a grid world environment. The assignment required implementing and comparing different search strategies including uninformed search algorithms (BFS, DFS) and informed search algorithms (A*, Beam Search, IDA*).

## What This Demonstrates

This project demonstrates:

- **Uninformed Search Algorithms**: Implementation of Breadth-First Search (BFS) and Depth-First Search (DFS) for pathfinding in grid worlds
- **Informed Search Algorithms**: Implementation of A* search with multiple heuristics (Manhattan and Euclidean distance)
- **Advanced Search Techniques**: 
  - Beam Search (memory-efficient variant of A*)
  - Iterative Deepening A* (IDA*) for optimal pathfinding with bounded memory
- **Algorithm Analysis**: Comparison of different search strategies in terms of path cost, nodes expanded, and frontier size
- **Visualization**: Animated visualization of search expansion and final paths

## What Was Provided

The following components were provided at the start of the assignment:

- **`utils/utils.py`**: Utility functions including:
  - `cost()`: Calculates movement cost based on terrain type
  - `expand()`: Generates valid neighboring cells for a given position
  - `visualize_expanded()`: Creates animated visualization of search expansion
  - `visualize_path()`: Visualizes the final computed path
  - Enums: `PathPlanMode`, `Heuristic`, `Environment`
  
- **`run_experiments.py`**: Command-line interface for running pathfinding experiments with various options (originally provided as `main.py`)

- **`worlds/`**: Directory containing 4 pre-generated grid world files:
  - `world_1.npy` through `world_4.npy`
  - Each world contains different terrain configurations (flatland, ponds, valleys, mountains)

- **Function signatures and docstrings**: The structure and expected behavior of functions to implement

## What I Contributed

I implemented the core search algorithms in `search_algorithms.py`:

### 1. **Graph Traversal Search** (`graph_traversal_search`)
   - Implements both BFS and DFS using a unified approach
   - Uses a frontier list with different pop strategies (index 0 for BFS, index -1 for DFS)
   - Tracks expanded nodes and frontier sizes for analysis
   - Reconstructs path using parent pointers

### 2. **Heuristic Path Search** (`heuristic_path_search`)
   - Implements A* algorithm with priority queue for optimal pathfinding
   - Supports both Manhattan and Euclidean distance heuristics
   - Includes Beam Search variant that limits frontier size to specified width
   - Tracks path costs, expanded nodes, and frontier sizes

### 3. **Iterative Deepening Search** (`iterative_deepening_search` and `__bounded_dfs_helper`)
   - Implements Iterative Deepening A* for memory-efficient optimal search
   - Uses depth-first search with cost bounds that increase iteratively
   - Calculates next bound when goal is not found within current bound
   - Supports both Manhattan and Euclidean heuristics

### 4. **Pathfinding Experiment Runner** (`run_pathfinding_experiment`)
   - Orchestrates testing of different algorithms on grid worlds
   - Calculates and reports path metrics (length, cost, nodes expanded, max frontier size)
   - Integrates with visualization system

## How to Run

### Prerequisites

```bash
pip install numpy matplotlib
```

### Basic Usage

```bash
python run_experiments.py <world_path> <world_id> [options]
```

### Arguments

- **`world_path`** (required): Directory containing the world `.npy` files
  - Example: `worlds`
  
- **`world_id`** (required): Which world to test (1-4)
  - Example: `1`, `2`, `3`, or `4`

### Options

#### `-e` (Heuristic/Search Mode)
Specifies which search algorithms and heuristics to use:

- **`-e 0`** or omitted: Uninformed search (DFS and BFS)
- **`-e 1`**: A* and Beam Search with Manhattan heuristic
- **`-e 2`**: A* and Beam Search with Euclidean heuristic
- **`-e 3`**: IDA* with Manhattan heuristic
- **`-e 4`**: IDA* with Euclidean heuristic

#### `-b` (Beam Width)
Specifies the beam width for beam search (only used with `-e 1` or `-e 2`):

- **Default**: 100
- **Example**: `-b 50` (limits frontier to top 50 nodes)

#### `-a` (Animation)
Controls visualization:

- **`-a 0`** or omitted: No visualization
- **`-a 1`**: Animated visualization of expanded nodes
- **`-a 2`**: Static visualization of final path only

### Example Commands

```bash
# Run BFS and DFS on world 1
python run_experiments.py worlds 1

# Run A* with Manhattan heuristic on world 2
python run_experiments.py worlds 2 -e 1

# Run A* and Beam Search with Euclidean heuristic, beam width 50, with animation
python run_experiments.py worlds 3 -e 2 -b 50 -a 1

# Run IDA* with Manhattan heuristic on world 4
python run_experiments.py worlds 4 -e 3

# Visualize final path only
python run_experiments.py worlds 1 -e 1 -a 2
```

## Output

The program outputs:

- **Search mode**: Which algorithm was used
- **Path length**: Number of cells in the path
- **Path cost**: Total cost of the path (sum of terrain costs)
- **Number of expanded states**: Total nodes explored
- **Max frontier size**: Maximum number of nodes in frontier at any point

Example output:
```
Testing world 1
Mode: A_STAR
Path length: 120
Path cost: 350
Number of expanded states: 450
Max frontier size: 200
```

## Grid World Terrain

The grid worlds contain four terrain types with different movement costs:

- **Flatland** (green): Cost = 3
- **Pond** (blue): Cost = 2 (cheapest terrain)
- **Valley** (orange): Cost = 5
- **Mountain** (gray): Cost = ∞ (impassable)

## Project Structure

```
ai-hw-1/
├── README.md                 # This file
├── run_experiments.py        # Command-line interface
├── search_algorithms.py      # Core search algorithm implementations
├── utils/
│   └── utils.py             # Provided utility functions
└── worlds/
    ├── world_1.npy          # Grid world test cases
    ├── world_2.npy
    ├── world_3.npy
    └── world_4.npy
```

## Algorithm Details

### Uninformed Search
- **BFS**: Explores level by level, guarantees shortest path length (but not necessarily lowest cost)
- **DFS**: Explores deeply first, may find longer paths but uses less memory

### A* Search
- Uses `f(n) = g(n) + h(n)` where:
  - `g(n)`: Actual cost from start to node n
  - `h(n)`: Heuristic estimate from node n to goal
- **Manhattan**: `|x1-x2| + |y1-y2|` (L1 distance)
- **Euclidean**: `√((x1-x2)² + (y1-y2)²)` (L2 distance)
- Guarantees optimal path when heuristic is admissible

### Beam Search
- Variant of A* that limits frontier to top-k nodes by f-value
- Reduces memory usage at cost of optimality guarantee

### IDA*
- Combines benefits of A* and iterative deepening
- Memory-efficient (O(bd) space vs A*'s O(b^d))
- Guarantees optimality with admissible heuristics

## Notes

- The start and goal positions are predefined for each world
- Mountains are completely impassable (infinite cost)
- The search algorithms handle 8-directional movement (including diagonals)
- Visualization requires matplotlib and may take time for large expansions

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
