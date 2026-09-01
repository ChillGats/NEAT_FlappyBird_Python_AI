import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import random

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.fc(x) * 2.0
    
def set_weights(model, weights):
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    torch.nn.utils.vector_to_parameters(weights_tensor, model.parameters())

def random_genome():
    return torch.nn.utils.parameters_to_vector(SimpleNN().parameters()).detach().numpy()

EVAL_MODEL = SimpleNN()

def evaluate(weights, render=False, steps=500):
    set_weights(EVAL_MODEL, weights)
    env = gym.make("Pendulum-v1", render_mode="human" if render else None)
    obs, _ = env.reset()
    total_reward = 0
    max_y = -1.0 

    for _ in range(steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = EVAL_MODEL(obs_tensor).numpy()
        obs, reward, done, truncated, _ = env.step(action)
        if obs[1] > max_y: max_y = obs[1]
        total_reward += reward
        if render:
            print(f"\r💪 Torque: {action[0]:.2f} | Y: {obs[1]:.2f}    ", end="")
        if done or truncated: break
    env.close()
    return total_reward + (max_y * 150)

def mutate(genome, mutation_rate, mutation_strength):
    new_genome = genome.copy()
    for i in range(len(new_genome)):
        if np.random.rand() < mutation_rate:
            new_genome[i] += np.random.normal(0, mutation_strength)
    return new_genome

# === CONFIGURATION PATIENTE ===
POP_SIZE = 80
GENERATIONS = 150 # On donne plus de temps
ELITE_SIZE = 8

population = [random_genome() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    # DÉCROISSANCE LINÉAIRE : Beaucoup plus stable
    # La force diminue très lentement de 0.1 à 0.01
    current_strength = max(0.01, 0.1 - (gen * 0.09 / GENERATIONS))
    
    scores = [evaluate(ind) for ind in population]
    best_score = max(scores)
    
    print(f"\r🧬 Gen {gen}/{GENERATIONS} | Mut: {current_strength:.3f} | Top: {best_score:.1f}    ", end="")

    sorted_indices = np.argsort(scores)[-ELITE_SIZE:]
    elites = [population[i] for i in sorted_indices]
    new_population = list(elites)
    while len(new_population) < POP_SIZE:
        p1, p2 = random.sample(elites, 2)
        mask = np.random.rand(len(p1)) < 0.5
        child = np.where(mask, p1, p2)
        child = mutate(child, 0.2, current_strength)
        new_population.append(child)
    population = new_population

print("\n\n🎮 Test Final...")
final_scores = [evaluate(ind) for ind in population]
evaluate(population[np.argmax(final_scores)], render=True, steps=500)
