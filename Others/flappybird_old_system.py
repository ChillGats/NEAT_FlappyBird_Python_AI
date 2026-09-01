import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import time
import os
import json

# === PARAMÈTRES SIMULATION ===
SETTINGS = {
    "SCREEN_WIDTH": 1000,
    "SCREEN_HEIGHT": 600,
    "PIPE_WIDTH": 100,
    "PIPE_HEIGHT": 500,
    "PIPE_GAP": 175,
    "PIPE_VELOCITY": 4, 
    "BIRD_SIZE": 30, 
    "GRAVITY": 0.5,
    "FLAP_STRENGTH": -10,

    "HIDDEN_SIZE": 32,
    "POP_SIZE": 25,
    "GENERATIONS": 50,
    "MAX_STEPS": 5000,
    "MUTATION_STD": 0.1,
    "MUTATION_RATE": 0.25,

    "FPS": 60,
    "VISUAL_DELAY": 10
}

best_model = None
best_model_ever = None
best_score_ever = float('-inf')

GENERATIONS_FOLDER = "models/generations"
os.makedirs(GENERATIONS_FOLDER, exist_ok=True)

CURRENT_DIR = os.path.dirname(__file__)
ASSETS_PATH = os.path.join(CURRENT_DIR,"..", "images")

# === CLASSES ===
class FlappyBirdGame:
    def __init__(self, render_mode="human"):
        self.settings = SETTINGS
        self.SCREEN_WIDTH = self.settings["SCREEN_WIDTH"]
        self.SCREEN_HEIGHT = self.settings["SCREEN_HEIGHT"]
        self.PIPE_WIDTH = self.settings["PIPE_WIDTH"]
        self.PIPE_HEIGHT = self.settings["PIPE_HEIGHT"]
        self.PIPE_GAP = self.settings["PIPE_GAP"]
        self.PIPE_VELOCITY = self.settings["PIPE_VELOCITY"]
        self.BIRD_SIZE = self.settings["BIRD_SIZE"]
        self.GRAVITY = self.settings["GRAVITY"]
        self.FLAP_STRENGTH = self.settings["FLAP_STRENGTH"]
        self.mg = 180
        self.time_rate = SETTINGS["FPS"]

        self.render_mode = render_mode
        self.screen = None
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird")

            self.pipe_img = pygame.image.load(os.path.join(ASSETS_PATH, "pipe_img.png")).convert_alpha()
            self.pipe_img = pygame.transform.scale(self.pipe_img, (325, 600))
            self.pipe_img_flipped = pygame.transform.flip(self.pipe_img, False, True)

            self.bird_img = pygame.image.load(os.path.join(ASSETS_PATH, "bird.png")).convert_alpha()
            self.bird_img = pygame.transform.scale(self.bird_img, (self.BIRD_SIZE * 2, self.BIRD_SIZE * 1.5))

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.next_pipe_id = 0
        self.pipe_pairs = [self.create_pipe_pair()]
        self.score = 0
        self.passed_pipes = set()
        self.done = False
        return self.get_obs()

    def create_pipe_pair(self):
        gap_y = random.randint(50, self.SCREEN_HEIGHT - 50 - self.PIPE_GAP)
        top_pipe = pygame.Rect(self.SCREEN_WIDTH, 0, self.PIPE_WIDTH, gap_y)
        bottom_pipe = pygame.Rect(self.SCREEN_WIDTH, gap_y + self.PIPE_GAP, self.PIPE_WIDTH,
                                  self.SCREEN_HEIGHT - (gap_y + self.PIPE_GAP))
        pipe_id = self.next_pipe_id
        self.next_pipe_id += 1
        return (top_pipe, bottom_pipe, pipe_id)

    def update_pipes(self, bird_rect, draw=False, fps_augmented=False):
        reward = 0
        new_pipe_needed = False
        self.pipe_vel = self.PIPE_VELOCITY * 2 if fps_augmented else self.PIPE_VELOCITY

        for i, (top_pipe, bottom_pipe, pipe_id) in enumerate(self.pipe_pairs):
            top_pipe.x -= self.pipe_vel
            bottom_pipe.x -= self.pipe_vel

            if draw and self.screen:
                top_x = top_pipe.centerx - self.pipe_img.get_width() // 2 - 6
                top_y = top_pipe.bottom - self.pipe_img_flipped.get_height() + self.mg
                bottom_x = bottom_pipe.centerx - self.pipe_img.get_width() // 2 - 6
                bottom_y = bottom_pipe.top - self.mg

                self.screen.blit(self.pipe_img, (top_x, top_y))
                self.screen.blit(self.pipe_img_flipped, (bottom_x, bottom_y))

            if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                self.done = True
                reward = -3

            # --- CORRECTION DU BUG DE SCORE ---
            if top_pipe.right < bird_rect.left:
                if pipe_id not in self.passed_pipes:  # Vérifie que le tuyau n'a pas DÉJÀ été compté
                    self.score += 1
                    reward += 5   # Belle récompense pour l'IA
                    self.passed_pipes.add(pipe_id)

            if top_pipe.centerx < self.SCREEN_WIDTH // 2 and i == len(self.pipe_pairs) - 1:
                new_pipe_needed = True

        if new_pipe_needed:
            self.pipe_pairs.append(self.create_pipe_pair())

        self.pipe_pairs = [(top, bottom, pipe_id) for (top, bottom, pipe_id) in self.pipe_pairs if top.right > 0]
        return reward

    def draw_pipes(self):
        if not self.screen:
            return

        for top_pipe, bottom_pipe, _ in self.pipe_pairs:
            top_x = top_pipe.centerx - self.pipe_img.get_width() // 2 - 6
            top_y = top_pipe.bottom - self.pipe_img_flipped.get_height() + self.mg
            bottom_x = bottom_pipe.centerx - self.pipe_img.get_width() // 2 - 6
            bottom_y = bottom_pipe.top - self.mg

            self.screen.blit(self.pipe_img, (top_x, top_y))
            self.screen.blit(self.pipe_img_flipped, (bottom_x, bottom_y))

    def get_obs(self):
        next_pipe = None
        for (top, bottom, _) in self.pipe_pairs:
            if top.right >= 50:
                next_pipe = (top, bottom)
                break

        gap_top = next_pipe[0].height
        gap_bottom = next_pipe[1].top

        if next_pipe:
            pipe_x_dist = next_pipe[0].left - 50
            pipe_y_center = next_pipe[0].height + self.PIPE_GAP / 2
        else:
            pipe_x_dist = 0
            pipe_y_center = self.SCREEN_HEIGHT / 2

        return [
            self.bird_y / self.SCREEN_HEIGHT,
            self.bird_velocity / 10.0,
            pipe_x_dist / self.SCREEN_WIDTH,
            (pipe_y_center - self.bird_y) / self.SCREEN_HEIGHT,
            gap_top / self.SCREEN_HEIGHT,
            gap_bottom / self.SCREEN_HEIGHT
        ]

    def step(self, action):
        if self.done:
            return self.get_obs(), 0, True, {}

        if action == 1:
            self.bird_velocity = self.FLAP_STRENGTH

        self.bird_velocity += self.GRAVITY
        self.bird_y += self.bird_velocity

        bird_rect = pygame.Rect(50, int(self.bird_y), self.BIRD_SIZE, self.BIRD_SIZE)
        if self.screen:
            self.screen.blit(self.bird_img, (bird_rect.x - 20, bird_rect.y - 20))
        reward = self.update_pipes(bird_rect)

        if self.bird_y > self.SCREEN_HEIGHT or self.bird_y < 0:
            self.done = True
            reward = -2

        return self.get_obs(), reward + 0.1, self.done, {}

    def run(self):
        running = True
        self.reset()
        pygame.time.wait(500)
        speed_multiplier = 1

        while running:
            self.clock.tick(SETTINGS["FPS"])
            action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        speed_multiplier = max(1, speed_multiplier // 2)
                    elif event.key == pygame.K_RIGHT:
                        speed_multiplier *= 2
                    elif event.key == pygame.K_UP:
                        speed_multiplier += 1
                    elif event.key == pygame.K_DOWN:
                        speed_multiplier = max(1, speed_multiplier - 1)

            for _ in range(speed_multiplier):
                if action == 1:
                    self.bird_velocity = self.FLAP_STRENGTH
                    action = 0

                self.bird_velocity += self.GRAVITY
                self.bird_y += self.bird_velocity
                bird_rect = pygame.Rect(50, int(self.bird_y), self.BIRD_SIZE, self.BIRD_SIZE)

                reward = self.update_pipes(bird_rect)
                if self.bird_y < 0 or self.bird_y > self.SCREEN_HEIGHT:
                    self.done = True
                    break

            if self.bird_y < 0 or self.bird_y > self.SCREEN_HEIGHT:
                self.done = True

            self.screen.fill((135, 206, 235))
            self.draw_pipes()
            angle = -self.bird_velocity * 3
            rotated_bird = pygame.transform.rotate(self.bird_img, angle)
            self.screen.blit(rotated_bird, (bird_rect.x - 20, bird_rect.y - 10))

            font = pygame.font.Font(None, 36)
            text = font.render(f"Score: {self.score}", True, (0, 0, 0))
            self.screen.blit(text, (10, 10))
            speed_text = font.render(f"Speed: {self.time_rate}", True, (0, 0, 0))
            self.screen.blit(speed_text, (10, 50))
            pygame.display.flip()

            if self.done:
                pygame.time.wait(1000)
                self.reset()
                self.done = False


class NeuralNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=None):
        super().__init__()
        hidden_size = hidden_size if hidden_size else SETTINGS["HIDDEN_SIZE"]
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# === NOUVEAUX OPÉRATEURS GÉNÉTIQUES ===

def crossover(parent1, parent2):
    """Mélange les poids de deux réseaux de neurones (50/50)."""
    child = NeuralNet(hidden_size=SETTINGS["HIDDEN_SIZE"])
    state1 = parent1.state_dict()
    state2 = parent2.state_dict()
    child_state = {}
    
    with torch.no_grad():
        for key in state1:
            # Crée un masque aléatoire de 0 et 1 pour choisir quel parent donne quel neurone
            mask = torch.rand_like(state1[key]) > 0.5
            # Combine les gènes
            child_state[key] = torch.where(mask, state1[key], state2[key])
            
    child.load_state_dict(child_state)
    return child

def mutate(model, mutation_rate=0.2, std=None):
    """Ne mute que 20% des poids, pour ne pas détruire l'apprentissage."""
    std = std if std else SETTINGS["MUTATION_STD"]
    new_model = NeuralNet(hidden_size=SETTINGS["HIDDEN_SIZE"])
    new_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param in new_model.parameters():
            # Choisit aléatoirement 20% des poids à muter
            mask = (torch.rand_like(param) < mutation_rate).float()
            noise = torch.randn_like(param) * std
            # N'ajoute le bruit qu'aux poids sélectionnés
            param.add_(mask * noise)
            
    return new_model


def evolutionary_training():
    global best_model_ever, best_score_ever
    population = [NeuralNet(hidden_size=SETTINGS["HIDDEN_SIZE"]) for _ in range(SETTINGS["POP_SIZE"])]
    
    # On garde les 5 meilleurs à chaque fois (20% de la population de 25)
    ELITE_SIZE = 5 

    for gen in range(SETTINGS["GENERATIONS"]):
        pygame.init()
        current_seed = random.randint(0, 999999)

        scores = []
        for model in population:
            random.seed(current_seed)
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            scores.append(evaluate(model))

        # --- NOUVELLE SÉLECTION (LE TOURNOI) ---
        # On trie la population du meilleur au pire score
        sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        
        # On extrait les X meilleurs (les Élites)
        elites = [p[1] for p in sorted_pairs[:ELITE_SIZE]]
        
        best_score = sorted_pairs[0][0]
        avg_score = sum(scores) / len(scores)

        print(f"Génération {gen + 1:02d}/{SETTINGS['GENERATIONS']} | Moy: {avg_score:6.2f} | Top: {best_score:6.2f}")

        # Sauvegarde du champion absolu
        if best_score > best_score_ever:
            best_score_ever = best_score
            best_model_ever = elites[0]

        save_generation(population, gen + 1, current_seed)

        # --- REPRODUCTION ---
        # La nouvelle génération commence avec nos élites intactes (pour ne jamais régresser)
        new_population = list(elites)

        # On remplit le reste de la population avec des enfants
        while len(new_population) < SETTINGS["POP_SIZE"]:
            # On choisit 2 parents au hasard parmi l'élite
            p1, p2 = random.sample(elites, 2)
            
            # Croisement
            child = crossover(p1, p2)
            
            # Mutation (Rate = 20% des gènes)
            child = mutate(child, mutation_rate=SETTINGS["MUTATION_RATE"], std=SETTINGS["MUTATION_STD"])
            
            new_population.append(child)

        population = new_population

    random.seed() 
    print("\n✅ Entraînement terminé !")
    return best_model_ever


def evaluate(model):
    game = FlappyBirdGame(render_mode=None)
    obs = game.reset()
    total_reward = 0
    steps = 0
    while not game.done and steps < SETTINGS["MAX_STEPS"]:
        x = torch.tensor(obs, dtype=torch.float32)
        output = model(x).item()
        action = 1 if output > 0.5 else 0
        obs, reward, done, _ = game.step(action)
        total_reward += reward
        steps += 1
    return total_reward

def save_generation(models, generation, seed):
    gen_folder = os.path.join(GENERATIONS_FOLDER, f"gen_{generation:03d}")
    os.makedirs(gen_folder, exist_ok=True)
    with open(os.path.join(gen_folder, "seed.json"), "w") as f:
        json.dump({"seed": seed}, f)
    for i, model in enumerate(models):
        filename = os.path.join(gen_folder, f"model_{i:03d}.pth")
        torch.save(model.state_dict(), filename)

def load_generation_models(generation):
    gen_folder = os.path.join(GENERATIONS_FOLDER, f"gen_{generation:03d}")
    if not os.path.exists(gen_folder):
        print(f"❌ Génération {generation} non trouvée.")
        return None, None
    with open(os.path.join(gen_folder, "seed.json"), "r") as f:
        data = json.load(f)
        seed = data.get("seed", None)
    model_files = sorted([f for f in os.listdir(gen_folder) if f.endswith(".pth")])
    models = []
    for f in model_files:
        model = NeuralNet()
        model.load_state_dict(torch.load(os.path.join(gen_folder, f)))
        models.append(model)
    return models, seed

def visualize_generation():
    generation = int(input("Quelle génération visualiser ? "))
    models, seed = load_generation_models(generation)
    if models is None: return

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    base_game = FlappyBirdGame(render_mode="human")
    game_clock = pygame.time.Clock()

    birds = []
    for model in models:
        bird = {
            "y": base_game.SCREEN_HEIGHT // 2,
            "velocity": 0,
            "alive": True,
            "model": model,
            "color": (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)),
            "score": 0,
            "passed_pipes": set()
        }
        birds.append(bird)

    steps = 0
    done = False
    game_clock.tick(SETTINGS["FPS"])
    time_rate = SETTINGS["FPS"]
    
    speed_multiplier = 1
    while not done:
        game_clock.tick(SETTINGS["FPS"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    speed_multiplier = max(1, speed_multiplier // 2)
                elif event.key == pygame.K_RIGHT:
                    speed_multiplier = min(960, speed_multiplier * 2)
                elif event.key == pygame.K_UP:
                    speed_multiplier = min(960, speed_multiplier + 1)
                elif event.key == pygame.K_DOWN:
                    speed_multiplier = max(1, speed_multiplier - 1)

        base_game.screen.fill((135, 206, 235))
        dummy_rect = pygame.Rect(50, 0, base_game.BIRD_SIZE, base_game.BIRD_SIZE)

        for step in range(speed_multiplier):
            draw = step == speed_multiplier - 1
            base_game.update_pipes(dummy_rect, draw=draw, fps_augmented=True)

            all_dead = True
            for bird in birds:
                if not bird["alive"]:
                    continue

                obs = [
                    bird["y"] / base_game.SCREEN_HEIGHT,
                    bird["velocity"] / 10.0,
                    base_game.pipe_pairs[0][0].left / base_game.SCREEN_WIDTH,
                    ((base_game.pipe_pairs[0][0].height + base_game.PIPE_GAP / 2) - bird["y"]) / base_game.SCREEN_HEIGHT,
                    base_game.pipe_pairs[0][0].height / base_game.SCREEN_HEIGHT,
                    base_game.pipe_pairs[0][1].top / base_game.SCREEN_HEIGHT
                ]

                x = torch.tensor(obs, dtype=torch.float32)
                output = bird["model"](x).item()
                action = 1 if output > 0.5 else 0

                if action == 1:
                    bird["velocity"] = base_game.FLAP_STRENGTH
                bird["velocity"] += base_game.GRAVITY
                bird["y"] += bird["velocity"]

                bird_rect = pygame.Rect(50, int(bird["y"]), base_game.BIRD_SIZE, base_game.BIRD_SIZE)

                angle = -base_game.bird_velocity * 3
                rotated_bird = pygame.transform.rotate(base_game.bird_img, angle)
                if draw:
                    base_game.screen.blit(rotated_bird, (bird_rect.x - 20, bird_rect.y - 10))

                for i, (top_pipe, bottom_pipe, pipe_id) in enumerate(base_game.pipe_pairs):
                    if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                        bird["alive"] = False
                        break
                    if bird["y"] < 0 or bird["y"] > base_game.SCREEN_HEIGHT:
                        bird["alive"] = False
                        break
                    if top_pipe.right < bird_rect.left and pipe_id not in bird["passed_pipes"]:
                        bird["score"] += 1
                        bird["passed_pipes"].add(pipe_id)

                if bird["alive"]:
                    all_dead = False

            if all_dead:
                break

        # Display scores for alive birds
        font = pygame.font.Font(None, 28)
        y_offset = 10
        for i, bird in enumerate(birds):
            if bird["alive"]:
                text = font.render(f"Bird {i + 1}: {bird['score']}", True, (0, 0, 0))
                base_game.screen.blit(text, (10, y_offset))
                y_offset += 30

        speed_text = font.render(f"Speed: {speed_multiplier}x", True, (0, 0, 0))
        base_game.screen.blit(speed_text, (10, y_offset))

        pygame.display.flip()
        steps += 1
        if all_dead:
            done = True

            obs = [
                bird["y"] / base_game.SCREEN_HEIGHT,
                bird["velocity"] / 10.0,
                base_game.pipe_pairs[0][0].left / base_game.SCREEN_WIDTH,
                ((base_game.pipe_pairs[0][0].height + base_game.PIPE_GAP / 2) - bird["y"]) / base_game.SCREEN_HEIGHT,
                base_game.pipe_pairs[0][0].height / base_game.SCREEN_HEIGHT,
                base_game.pipe_pairs[0][1].top / base_game.SCREEN_HEIGHT
            ]

            x = torch.tensor(obs, dtype=torch.float32)
            output = bird["model"](x).item()
            action = 1 if output > 0.5 else 0

            if action == 1: bird["velocity"] = base_game.FLAP_STRENGTH
            bird["velocity"] += base_game.GRAVITY
            bird["y"] += bird["velocity"]

            bird_rect = pygame.Rect(50, int(bird["y"]), base_game.BIRD_SIZE, base_game.BIRD_SIZE)

            angle = -base_game.bird_velocity * 3
            rotated_bird = pygame.transform.rotate(base_game.bird_img, angle)
            base_game.screen.blit(rotated_bird, (bird_rect.x - 20, bird_rect.y - 10))

            # Check collisions and score for all pipes
            for i, (top_pipe, bottom_pipe, pipe_id) in enumerate(base_game.pipe_pairs):
                if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                    bird["alive"] = False
                    break
                if bird["y"] < 0 or bird["y"] > base_game.SCREEN_HEIGHT:
                    bird["alive"] = False
                    break
                if top_pipe.right < bird_rect.left and pipe_id not in bird["passed_pipes"]:
                    bird["score"] += 1
                    bird["passed_pipes"].add(pipe_id)

            if not bird["alive"]:
                continue

            all_dead = False

        # Display scores for alive birds
        font = pygame.font.Font(None, 28)
        y_offset = 10
        for i, bird in enumerate(birds):
            if bird["alive"]:
                text = font.render(f"Bird {i + 1}: {bird['score']}", True, (0, 0, 0))
                base_game.screen.blit(text, (10, y_offset))
                y_offset += 30

        speed_text = font.render(f"FPS: {time_rate}", True, (0, 0, 0))
        base_game.screen.blit(speed_text, (10, y_offset))

        pygame.display.flip()
        steps += 1
        if all_dead: done = True

    print("✅ Visualisation terminée.")
    random.seed()
    pygame.time.wait(1500)

def play_best_ai():
    global best_model_ever
    if best_model_ever is None:
        print("❌ Aucun modèle chargé ou entraîné.")
        return

    game = FlappyBirdGame(render_mode="human")
    game_clock = pygame.time.Clock()
    obs = game.reset()
    running = True
    speed_multiplier = 1
    
    while running:
        game_clock.tick(SETTINGS["FPS"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    speed_multiplier = max(1, speed_multiplier // 2)
                elif event.key == pygame.K_RIGHT:
                    speed_multiplier = min(960, speed_multiplier * 2)
                elif event.key == pygame.K_UP:
                    speed_multiplier = min(960, speed_multiplier + 1)
                elif event.key == pygame.K_DOWN:
                    speed_multiplier = max(1, speed_multiplier - 1)

        done = False
        for _ in range(speed_multiplier):
            x = torch.tensor(np.array(obs), dtype=torch.float32)
            output = best_model_ever(x).item()
            action = 1 if output > 0.5 else 0
            obs, reward, done, _ = game.step(action)
            if done:
                break

        game.screen.fill((135, 206, 235))
        game.draw_pipes()
        bird_rect = pygame.Rect(50, int(game.bird_y), game.BIRD_SIZE, game.BIRD_SIZE)

        angle = -game.bird_velocity * 3
        rotated_bird = pygame.transform.rotate(game.bird_img, angle)
        game.screen.blit(rotated_bird, (bird_rect.x -20 , bird_rect.y -10))

        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {game.score}", True, (0, 0, 0))
        game.screen.blit(text, (10, 10))
        speed_text = font.render(f"Speed: {speed_multiplier}x", True, (0, 0, 0))
        game.screen.blit(speed_text, (10, 50))
        
        text_output = font.render(f"Output: {output:.2f}", True, (255, 0, 0))
        game.screen.blit(text_output, (10, 90))
        
        pygame.display.flip()

        if done:
            pygame.time.wait(1500)
            obs = game.reset()

def customize_settings():
    print("\n--- Personnalisation des paramètres d'évolution ---")
    for key in ["POP_SIZE", "HIDDEN_SIZE", "GENERATIONS", "MAX_STEPS", "MUTATION_STD", "MUTATION_RATE"]:
        current = SETTINGS[key]
        new_value = input(f"{key} (actuel: {current}) : ")
        if new_value:
            if "." in new_value:
                SETTINGS[key] = float(new_value)
            else:
                SETTINGS[key] = int(new_value)
    print("✅ Paramètres mis à jour.")

def Menu():
    print("\n=== MENU PRINCIPAL ===")
    print("1 - jouer une partie")
    print("2 - entrainer une population IA")
    print("3 - faire jouer la meilleure IA")
    print("4 - personnaliser les paramètres d'évolution")
    print("5 - visualiser une génération complète")
    print("6 - sauvegarder la meilleure IA")
    print("7 - charger la meilleure IA")
    print("10 - supprimer un modèle existant")
    print("0 - quitter le programme")

    choice = input("Choix : ")

    if choice == '1':
        game = FlappyBirdGame()
        game.run()
    elif choice == '2':
        approx_time = SETTINGS["GENERATIONS"] * SETTINGS["POP_SIZE"] * SETTINGS["MAX_STEPS"] // 30000
        confirm = input(f"L'entraînement peut prendre environ {approx_time:.1f} secondes. Continuer ? (o/n) ")
        if confirm.lower() == 'o':
            evolutionary_training()
    elif choice == '3':
        play_best_ai()
    elif choice == '4':
        customize_settings()
    elif choice == '5':
        visualize_generation()
    elif choice == '6':
        global best_model_ever
        if best_model_ever is not None:
            name = input("Nom du modèle à sauvegarder (sans extension) : ")
            filename = f"models/{name}.pth"
            os.makedirs("models", exist_ok=True)
            torch.save(best_model_ever.state_dict(), filename)
            print(f"✅ Modèle sauvegardé sous {filename}.")
        else:
            print("❌ Aucun modèle à sauvegarder.")
    elif choice == '7':
        os.makedirs("models", exist_ok=True)
        files = [f for f in os.listdir("models") if f.endswith(".pth")]
        if not files:
            print("❌ Aucun modèle trouvé.")
        else:
            for i, file in enumerate(files):
                print(f"{i + 1} - {file}")
            idx = int(input("Numéro du modèle à charger : ")) - 1
            if 0 <= idx < len(files):
                best_model_ever = NeuralNet()
                best_model_ever.load_state_dict(torch.load(os.path.join("models", files[idx])))
                print(f"✅ Modèle {files[idx]} chargé.")
            else:
                print("❌ Numéro invalide.")
    elif choice == '10':
        os.makedirs("models", exist_ok=True)
        files = [f for f in os.listdir("models") if f.endswith(".pth")]
        if not files:
            print("❌ Aucun modèle à supprimer.")
        else:
            for i, file in enumerate(files):
                print(f"{i + 1} - {file}")
            idx = int(input("Numéro du modèle à supprimer : ")) - 1
            if 0 <= idx < len(files):
                os.remove(os.path.join("models", files[idx]))
                print(f"✅ Modèle {files[idx]} supprimé.")
            else:
                print("❌ Numéro invalide.")
    elif choice == '0':
        print("Fermeture du programme.")
        pygame.quit()
        sys.exit()
    else:
        print("Choix non reconnu.")

    Menu()

if __name__ == "__main__":
    Menu()
