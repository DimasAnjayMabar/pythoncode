import random
import numpy as np

# Parameter GA
POP_SIZE = 50
GEN_COUNT = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
DAYS = 30
SHIFTS = ["Pagi", "Sore", "Malam"]

# Data perawat (ID, Nama, Senioritas, Sertifikasi)
nurses = [
    (1, "Andi", "Senior", ["ICU"]),
    (2, "Budi", "Junior", []),
    (3, "Citra", "Senior", ["Bayi"]),
    (4, "Dewi", "Junior", []),
    (5, "Eko", "Senior", ["ICU", "Bayi"])
]

# Data bangsal yang membutuhkan sertifikasi
special_wards = {
    "ICU": 2,
    "Bayi": 3
}

# Inisialisasi populasi
def random_chromosome():
    schedule = []
    for day in range(DAYS):
        daily_schedule = []
        for shift in SHIFTS:
            assigned_nurses = random.sample(nurses, k=4)
            daily_schedule.append((day, shift, assigned_nurses))
        schedule.append(daily_schedule)
    return schedule

def initialize_population():
    return [random_chromosome() for _ in range(POP_SIZE)]

# Fungsi fitness
def evaluate_fitness(chromosome):
    penalty = 0
    assigned_nurses = {}
    
    for daily_schedule in chromosome:
        for day, shift, nurses_list in daily_schedule:
            for nurse in nurses_list:
                if (nurse[0], day) in assigned_nurses:
                    penalty += 50  # Penalti jika perawat bekerja dua shift dalam sehari
                assigned_nurses[(nurse[0], day)] = shift
                
                # Cek sertifikasi untuk bangsal khusus
                for ward, required_nurses in special_wards.items():
                    if len([n for n in nurses_list if ward in n[3]]) < required_nurses:
                        penalty += 100  # Penalti jika kurang perawat bersertifikasi
    
    return 10000 - penalty  # Skor dasar dikurangi penalti

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
        day = random.randint(0, DAYS - 1)
        shift = random.randint(0, len(SHIFTS) - 1)
        i, j = random.sample(range(len(nurses)), 2)
        chromosome[day][shift][2][i], chromosome[day][shift][2][j] = chromosome[day][shift][2][j], chromosome[day][shift][2][i]
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
    print("Best nurse schedule:", best_solution)
    print("Best fitness:", evaluate_fitness(best_solution))

if __name__ == "__main__":
    genetic_algorithm()
