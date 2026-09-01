# 🐦 Autonomous FlappyBird AI — Genetic Algorithm & Vectorized PyTorch

An autonomous FlappyBird AI agent powered by evolutionary neural networks and custom vectorized genetic operators implemented in **PyTorch**.

---

## 🔬 Architecture & First Principles

### 1. State Vector Formulation ($\mathbb{R}^6$)
At each simulation step, the environment feeds a 6-dimensional observation vector to the neural network:
1. ird_y: Normalized vertical position of the agent.
2. ird_vel_y: Vertical velocity of the agent.
3. 
ext_pipe_dist_x: Horizontal distance to the upcoming obstacle.
4. 
ext_pipe_top_y: Upper bound of the upcoming pipe gap.
5. 
ext_pipe_bottom_y: Lower bound of the upcoming pipe gap.
6. second_pipe_dist_x: Distance to the subsequent pipe for trajectory lookahead.

### 2. Neural Architecture
\mathbf{x} \in \mathbb{R}^6 \xrightarrow{\text{Linear}(6, 16)} \mathbf{h} \in \mathbb{R}^{16} \xrightarrow{\text{Tanh}} \mathbf{h}' \xrightarrow{\text{Linear}(16, 1)} z \xrightarrow{\text{Sigmoid}} y \in [0, 1]
- **Decision Rule**: Action = Jump ($) if  > 0.5$, else Coast ($).

### 3. Vectorized Genetic Operators
- **Uniform Tensor Crossover**: Generates offspring weights by blending parent genomes via vectorized boolean sampling without Python loop overhead:
  \mathbf{W}_{\text{child}} = \mathbf{M} \odot \mathbf{W}_{\text{P1}} + (1 - \mathbf{M}) \odot \mathbf{W}_{\text{P2}}, \quad \mathbf{M} \sim \text{Bernoulli}(0.5)
- **Gaussian Mutation with Bounded Perturbation**: Perturbs a fraction ($\alpha = \text{mutation\_rate}$) of parameters with zero-mean Gaussian noise $\mathcal{N}(0, \sigma^2)$ to prevent catastrophic forgetting.

---

## ⚡ Quick Start

### 1. Requirements & Installation
`ash
git clone https://github.com/ChillGats/NEAT_FlappyBird_Python_AI.git
cd NEAT_FlappyBird_Python_AI
pip install -r requirements.txt
`

### 2. Execution Modes
`ash
python main.py
`
From the interactive CLI menu:
- 1. Play: Manual human control.
- 2. Train: Run the genetic evolutionary loop over generations.
- 3. Watch Best: Benchmark the best-ever evolved model checkpoint.
- 4. Visualize Generation: Render all agents across a selected generation in real-time.

---

## 📁 Repository Structure
\\\
NEAT_FlappyBird_Python_AI/
├── main.py              # CLI entry point and menu router
├── game.py              # Custom Pygame simulation environment
├── model.py             # PyTorch NeuralNet definition & vectorized genetic operators
├── train.py             # Evolutionary loop, fitness evaluation & checkpoints
├── visualize.py         # Multi-agent generation visualizer
├── requirements.txt     # Dependencies (torch, pygame, numpy)
├── TRAINING.md          # In-depth hyperparameter tuning guide
└── Others/
    ├── flappybird_old_system.py
    └── gymnasium_neat_demo.py
\\\

---

## 📊 Benchmark & Persistence
- Generation checkpoints are automatically stored in generations/gen_XXX/ with state dictionaries and random seeds for strict scientific reproducibility.
- Top-performing models are serialized in models/ along with hyperparameter metadata JSON files.