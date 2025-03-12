import random
import numpy as np

# Parameter GA
POP_SIZE = 50
GEN_COUNT = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1

# Data bus (ID, Kapasitas, Maintenance Schedule)
buses = [
    (1, 40, ["Senin"]),
    (2, 50, ["Selasa"]),
    (3, 45, ["Rabu"]),
    (4, 55, ["Kamis"]),
    (5, 60, ["Jumat"])
]

# Data driver (ID, Nama, Jadwal Cuti)
drivers = [
    (1, "Andi", ["Senin"]),
    (2, "Budi", ["Rabu"]),
    (3, "Charlie", ["Jumat"]),
    (4, "Dani", ["Minggu"])
]

# Data kondektur (ID, Nama, Jadwal Cuti)
conductors = [
    (1, "Eko", ["Selasa"]),
    (2, "Fahmi", ["Kamis"]),
    (3, "Gilang", ["Sabtu"])
]

# Data rute (Kota Asal, Kota Tujuan, Jarak dalam km)
routes = [
    ("Jakarta", "Surabaya", 780),
    ("Bandung", "Semarang", 450),
    ("Surabaya", "Bali", 400),
    ("Medan", "Padang", 360)
]

# Inisialisasi populasi
def random_chromosome():
    return [(random.choice(buses), random.choice(routes), random.choice(drivers), random.choice(conductors)) for _ in range(len(routes))]

def initialize_population():
    return [random_chromosome() for _ in range(POP_SIZE)]

# Fungsi fitness
def evaluate_fitness(chromosome):
    profit = 0
    penalty = 0
    assigned_drivers = set()
    assigned_conductors = set()
    assigned_buses = set()
    
    for bus, route, driver, conductor in chromosome:
        if bus in assigned_buses or driver in assigned_drivers or conductor in assigned_conductors:
            penalty += 50  # Penalti jika ada konflik jadwal
        assigned_buses.add(bus)
        assigned_drivers.add(driver)
        assigned_conductors.add(conductor)
        profit += bus[1] * 100  # Kapasitas bus x tarif per penumpang
    
    return profit - penalty

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
        chromosome[i] = (random.choice(buses), random.choice(routes), random.choice(drivers), random.choice(conductors))
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
    print("Best bus schedule:", best_solution)
    print("Best fitness:", evaluate_fitness(best_solution))

if __name__ == "__main__":
    genetic_algorithm()
