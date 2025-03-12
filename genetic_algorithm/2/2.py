import random
import numpy as np

# Parameter GA
POP_SIZE = 50
GEN_COUNT = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1

# Data peti kemas (ID, Pelabuhan Tujuan, Berat, Jenis (0: biasa, 1: khusus))
containers = [
    (1, "Jakarta", 10, 0),
    (2, "Surabaya", 15, 0),
    (3, "Bali", 8, 1),
    (4, "Semarang", 12, 0),
    (5, "Makassar", 20, 0),
    (6, "Medan", 25, 1),
    (7, "Batam", 18, 0)
]

# Kapasitas maksimal kapal (dalam ton)
SHIP_CAPACITY = 100

# Inisialisasi populasi
def random_chromosome():
    return random.sample(containers, len(containers))

def initialize_population():
    return [random_chromosome() for _ in range(POP_SIZE)]

# Fungsi fitness
def evaluate_fitness(chromosome):
    total_weight = sum(c[2] for c in chromosome)
    penalty = 0
    if total_weight > SHIP_CAPACITY:
        penalty += (total_weight - SHIP_CAPACITY) * 10  # Penalti jika melebihi kapasitas
    
    # Urutan bongkar harus sesuai dengan jarak pelabuhan
    port_order = [c[1] for c in chromosome]
    if port_order != sorted(port_order):
        penalty += 50  # Penalti jika urutan salah
    
    return sum(c[2] for c in chromosome) - penalty

# Seleksi dengan Roulette Wheel
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return population[np.random.choice(len(population), p=selection_probs)]

# Order Crossover (OX1)
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        size = len(parent1)
        point1, point2 = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[point1:point2] = parent1[point1:point2]
        remaining = [item for item in parent2 if item not in child]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[idx]
                idx += 1
        return child
    return parent1

# Swap Mutation
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# Algoritma Genetika
def genetic_algorithm():
    population = initialize_population()
    for _ in range(GEN_COUNT):
        fitnesses = [evaluate_fitness(ind) for ind in population]
        new_population = []
        for _ in range(POP_SIZE // 2):
            parent1, parent2 = select(population, fitnesses), select(population, fitnesses)
            offspring = crossover(parent1, parent2)
            new_population.append(mutate(offspring))
        population = sorted(new_population, key=evaluate_fitness, reverse=True)[:POP_SIZE]
    
    best_solution = max(population, key=evaluate_fitness)
    print("Best container placement:", best_solution)
    print("Best fitness:", evaluate_fitness(best_solution))

if __name__ == "__main__":
    genetic_algorithm()
