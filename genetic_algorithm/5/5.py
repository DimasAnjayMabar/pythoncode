import random
import numpy as np

# Parameter GA
POP_SIZE = 50
GEN_COUNT = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1

# Data mesin produksi (ID, Barang yang dapat diproduksi, Jadwal Maintenance)
machines = [
    (1, ["A", "B", "C"], ["Senin"]),
    (2, ["B", "C", "D"], ["Selasa"]),
    (3, ["A", "D", "E"], ["Rabu"]),
    (4, ["C", "E", "F"], ["Kamis"])
]

# Data barang (ID, Waktu Produksi, Keuntungan per unit)
products = {
    "A": (4, 500),
    "B": (5, 600),
    "C": (3, 400),
    "D": (6, 700),
    "E": (2, 300),
    "F": (8, 900)
}

# Slot waktu produksi dalam 1 minggu (48 jam per mesin)
WEEKLY_HOURS = 48

# Inisialisasi populasi
def random_chromosome():
    schedule = []
    for machine in machines:
        machine_schedule = []
        available_hours = WEEKLY_HOURS
        while available_hours > 0:
            product = random.choice(machine[1])
            time_required = products[product][0]
            if available_hours - time_required >= 0:
                machine_schedule.append((product, time_required))
                available_hours -= time_required
            else:
                break
        schedule.append((machine[0], machine_schedule))
    return schedule

def initialize_population():
    return [random_chromosome() for _ in range(POP_SIZE)]

# Fungsi fitness
def evaluate_fitness(chromosome):
    total_profit = 0
    penalty = 0
    for machine_id, schedule in chromosome:
        used_hours = sum(time for _, time in schedule)
        if used_hours > WEEKLY_HOURS:
            penalty += (used_hours - WEEKLY_HOURS) * 50  # Penalti jika waktu produksi melebihi batas
        total_profit += sum(products[item][1] for item, _ in schedule)
    return total_profit - penalty

# Seleksi dengan Roulette Wheel
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return population[np.random.choice(len(population), p=selection_probs)]

# One-Point Crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

# Swap Mutation
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i = random.randint(0, len(chromosome) - 1)
        j = random.randint(0, len(chromosome[i][1]) - 1)
        new_product = random.choice([p for p in products if p not in chromosome[i][1]])
        chromosome[i][1][j] = (new_product, products[new_product][0])
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
    print("Best production schedule:", best_solution)
    print("Best fitness:", evaluate_fitness(best_solution))

if __name__ == "__main__":
    genetic_algorithm()
