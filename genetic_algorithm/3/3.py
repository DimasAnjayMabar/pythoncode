import random
import numpy as np

# Parameter GA
POP_SIZE = 50
GEN_COUNT = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
MAX_TRUCKS = 4

# Data barang (ID, Berat, Kota Tujuan, Dimensi)
items = [
    (1, 100, "Jakarta", "Medium"),
    (2, 200, "Surabaya", "Large"),
    (3, 150, "Bali", "Small"),
    (4, 300, "Bandung", "Large"),
    (5, 120, "Medan", "Medium"),
    (6, 250, "Makassar", "Small"),
    (7, 180, "Semarang", "Medium")
]

# Kapasitas maksimal per mobil (dalam kg)
TRUCK_CAPACITY = 500

# Inisialisasi populasi
def random_chromosome():
    return [(item, random.randint(1, MAX_TRUCKS)) for item in items]

def initialize_population():
    return [random_chromosome() for _ in range(POP_SIZE)]

# Fungsi fitness
def evaluate_fitness(chromosome):
    truck_loads = {i: 0 for i in range(1, MAX_TRUCKS + 1)}
    penalty = 0
    total_profit = 0
    
    for item, truck in chromosome:
        truck_loads[truck] += item[1]
        total_profit += item[1] * 10  # Profit = berat x tarif per kg
    
    for load in truck_loads.values():
        if load > TRUCK_CAPACITY:
            penalty += (load - TRUCK_CAPACITY) * 10
    
    return total_profit - penalty

# Seleksi dengan Roulette Wheel
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return population[np.random.choice(len(population), p=selection_probs)]

# Uniform Crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        child = []
        for i in range(len(parent1)):
            child.append(parent1[i] if random.random() < 0.5 else parent2[i])
        return child
    return parent1

# Swap Mutation
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i = random.randint(0, len(chromosome) - 1)
        new_truck = random.randint(1, MAX_TRUCKS)
        chromosome[i] = (chromosome[i][0], new_truck)
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
    print("Best truck assignment:", best_solution)
    print("Best fitness:", evaluate_fitness(best_solution))

if __name__ == "__main__":
    genetic_algorithm()
