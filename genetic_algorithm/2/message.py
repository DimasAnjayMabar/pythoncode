import random
from tabulate import tabulate

# Konstanta
TONASE_MAKSIMAL_KAPAL = 500  # Kapasitas maksimal kapal (dalam ton)
BIAYA_PER_TON = 100  # Biaya per ton untuk peti biasa
BIAYA_PER_TON_KHUSUS = 500  # Biaya per ton untuk peti khusus

# Struktur Peti Kemas (ID, Bobot, Jenis, Kota Tujuan)
PETI_KEMAS = [
    (1, 50, 'biasa', 'Jakarta'),
    (2, 30, 'biasa', 'Surabaya'),
    (3, 20, 'biasa', 'Makassar'),
    (4, 70, 'khusus', 'Medan'),
    (5, 60, 'biasa', 'Batam'),
    (6, 90, 'biasa', 'Bali'),
    (7, 40, 'khusus', 'Pontianak'),
    (8, 100, 'biasa', 'Ambon'),
    (9, 80, 'biasa', 'Manado')
]

# Inisialisasi Populasi
POPULASI_AWAL = 3
ITERASI = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

def generate_individual():
    """Membuat solusi awal dengan urutan acak, memastikan tidak melebihi tonase maksimal."""
    individual = random.sample(PETI_KEMAS, len(PETI_KEMAS))  # Urutan acak
    total_weight = 0
    solution = []
    for peti in individual:
        if total_weight + peti[1] <= TONASE_MAKSIMAL_KAPAL:
            solution.append(peti)
            total_weight += peti[1]
    return solution

def fitness(individual):
    """Menghitung fitness berdasarkan keuntungan total."""
    revenue = sum(peti[1] * (BIAYA_PER_TON_KHUSUS if peti[2] == 'khusus' else BIAYA_PER_TON) for peti in individual)
    return revenue

def selection(population):
    """Seleksi menggunakan roulette wheel selection."""
    total_fitness = sum(fitness(ind) for ind in population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind in population:
        current += fitness(ind)
        if current > pick:
            return ind

def crossover(parent1, parent2):
    """Order Crossover (OX1) untuk mempertahankan urutan unloading."""
    if random.random() > CROSSOVER_RATE:
        return parent1[:]

    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[point1:point2] = parent1[point1:point2]

    remaining = [p for p in parent2 if p not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[idx]
            idx += 1
    return child

def mutation(individual):
    """Swap mutation untuk eksplorasi solusi baru."""
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def print_population(population, title):
    """Mencetak populasi dalam bentuk tabel."""
    print(f"\n{title}")
    for i, ind in enumerate(population):
        table_data = [(peti[0], peti[1], peti[2], peti[3]) for peti in ind]
        headers = ["ID", "Bobot (Ton)", "Jenis", "Kota Tujuan"]
        print(f"Individu {i + 1} (Fitness: {fitness(ind)})")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

def genetic_algorithm():
    """Proses utama Algoritma Genetika."""
    population = [generate_individual() for _ in range(POPULASI_AWAL)]
    print_population(population, "Populasi Awal")

    for iteration in range(ITERASI):
        print(f"\n=== Iterasi {iteration + 1} ===")
        new_population = []
        for _ in range(POPULASI_AWAL // 2):
            parent1, parent2 = selection(population), selection(population)
            print_population([parent1, parent2], "Seleksi Orang Tua")
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            print_population([child1, child2], "Setelah Crossover")
            mutated_child1, mutated_child2 = mutation(child1), mutation(child2)
            print_population([mutated_child1, mutated_child2], "Setelah Mutasi")
            new_population.extend([mutated_child1, mutated_child2])

        print_population(new_population, "Populasi Setelah Mutasi")
        population = sorted(new_population, key=fitness, reverse=True)[:POPULASI_AWAL]
        print_population(population, f"Populasi Setelah Seleksi (Iterasi {iteration + 1})")
        print(f"Iterasi {iteration + 1}: Best Fitness = {fitness(population[0])}")

    return population[0]

# Menjalankan Algoritma Genetika
best_solution = genetic_algorithm()

# Menampilkan solusi terbaik dalam bentuk tabel
table_data = [(peti[0], peti[1], peti[2], peti[3]) for peti in best_solution]
headers = ["ID", "Bobot (Ton)", "Jenis", "Kota Tujuan"]
print("\nSolusi Terbaik:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))