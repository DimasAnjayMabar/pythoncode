import random
import numpy as np

# Parameter GA
POP_SIZE = 50
GEN_COUNT = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
NUM_GENERATORS = 7
PERIODS = 12
POWER_REQUIREMENT = 100
RESERVE_REQUIREMENT = 15

# Data pembangkit listrik
generators = [
    (1, 20, 2),
    (2, 15, 2),
    (3, 35, 1),
    (4, 40, 1),
    (5, 15, 1),
    (6, 15, 2),
    (7, 10, 1)
]

# Inisialisasi populasi def random_chromosome():
def random_chromosome():
    return [random.randint(1, PERIODS) for _ in range(NUM_GENERATORS)]

def initialize_population():
    return [random_chromosome() for _ in range(POP_SIZE)]

# Fungsi fitness
def evaluate_fitness(chromosome):
    power_supply = [0] * PERIODS
    for i, period in enumerate(chromosome):
        power_supply[period - 1] += generators[i][1]
    
    violations = sum(1 for power in power_supply if power < POWER_REQUIREMENT - RESERVE_REQUIREMENT)
    return -violations

# Seleksi dengan Roulette Wheel
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return population[np.random.choice(len(population), p=selection_probs)]

# Crossover One-Point
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, NUM_GENERATORS - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

# Mutasi
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, NUM_GENERATORS - 1)
        chromosome[idx] = random.randint(1, PERIODS)
    return chromosome

# Algoritma Genetika
def genetic_algorithm():
    population = initialize_population()
    for _ in range(GEN_COUNT):
        fitnesses = [evaluate_fitness(ind) for ind in population]
        new_population = []
        for _ in range(POP_SIZE // 2):
            parent1, parent2 = select(population, fitnesses), select(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])
        population = sorted(new_population, key=evaluate_fitness, reverse=True)[:POP_SIZE]
    
    best_solution = max(population, key=evaluate_fitness)
    print("Best maintenance schedule:", best_solution)
    print("Best fitness:", evaluate_fitness(best_solution))

if __name__ == "__main__":
    genetic_algorithm()
