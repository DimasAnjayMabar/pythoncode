import random
import numpy as np

# Parameter GA
POP_SIZE = 50
GEN_COUNT = 5
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05
SPECIAL_WEIGHT = 50000000  # 50jt Rupiah

# Konversi ke format mata uang Rupiah
def format_rupiah(amount):
    return f"Rp {amount:,.0f}".replace(",", ".")

# Data kapal dengan kapasitas maksimal
def initialize_expedition():
    return {
        "Ship1": {"id": 1, "name": "Ship1", "capacity": 70, "container": []},
        "Ship2": {"id": 2, "name": "Ship2", "capacity": 80, "container": []},
        "Ship3": {"id": 3, "name": "Ship3", "capacity": 90, "container": []}
    }

# Data tujuan dengan jarak (semakin kecil, semakin dekat. Jarak dalam km)
destinations = {
    1: {"id": 1, "destination": "Surabaya", "range": 760},
    2: {"id": 2, "destination": "Jakarta", "range": 400},
    3: {"id": 3, "destination": "Medan", "range": 1900},
    4: {"id": 4, "destination": "Manado", "range": 900},
    5: {"id": 5, "destination": "Ambon", "range": 800},
    6: {"id": 6, "destination": "Jayapura", "range": 2500},
    7: {"id": 7, "destination": "Batam", "range": 590}
}

# Data peti kemas (berat dalam ton, cost dalam Rp)
containers = {
    1: {"id": 1, "weight": 30, "isSpecial": True, "destination": ["Surabaya", 760], "cost": 20000000 + SPECIAL_WEIGHT},
    2: {"id": 2, "weight": 40, "isSpecial": False, "destination": ["Jakarta", 400], "cost": 30000000},
    3: {"id": 3, "weight": 50, "isSpecial": True, "destination": ["Medan", 1900], "cost": 40000000 + SPECIAL_WEIGHT}
}

# Fungsi inisialisasi target price yang dynamic agar meningkatkan kinerja fitness function
def get_initial_target_price(population):
    avg_cost = sum(evaluate_fitness(ind) for ind in population) / len(population) # Menghitung average cost dari initial population
    return avg_cost * 1.2 # Set 20% lebih tinggi dari avg cost untuk target price

# Inisialisasi populasi
def initialize_population():
    population = [] # Array population
    for _ in range(POP_SIZE): # Loop inisialisasi population
        expedition = initialize_expedition() # 1 expedisi = 1 populasi (berisi 3 kapal)
        container_list = list(containers.values()) # Mengambil list container
        random.shuffle(container_list) # Mengacak isi list container
        
        for container in container_list:
            possible_ships = [
                ship for ship in expedition.values() 
                    if sum(c["weight"] for c in ship["container"]) + container["weight"] <= ship["capacity"]] # Memilih kapal yang cukup untuk diisi container
            if possible_ships:
                chosen_ship = random.choice(possible_ships) # Memilih kapal yang sekiranya cukup
                chosen_ship["container"].append(container) # Memasukkan container ke dalam kapal
        
        population.append(expedition) # Menambah expedition ke dalam population 
    return population # Return population

# Fungsi fitness
def evaluate_fitness(expedition):
    total_cost = sum(container["cost"] for ship in expedition.values() for container in ship["container"]) # Menghitung total cost dalam satu kapal
    
    # Perhitungan container hilang akan diberi penalti
    assigned_containers = {container["id"] for ship in expedition.values() for container in ship["container"]} # Mengambil data container dari setiap kapal
    total_containers = set(containers.keys())  # Menghitung total container yang ada dalam setiap kapal
    missing_containers = total_containers - assigned_containers  # Menghitung total container hilang
    
    if missing_containers:
        penalty = len(missing_containers) * 1_000_000_000  # Pemberian penalti ketika container hilang
        total_cost += penalty # Penalti di tambah di cost pengiriman container
        print(f"[PENALTY] {len(missing_containers)} missing containers! Adding {format_rupiah(penalty)} penalty.") # Debug

    return total_cost # Return total cost

# Fungsi crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE: # Cek apakah iterasi populasi sekarang termasuk yang bisa di crossover atau tidak
        child = initialize_expedition() # Inisialisasi populasi baru untuk child
        for ship_name in parent1.keys(): # Loop per kapal per populasi atau parent
            if random.random() < 0.5: # Membagi container menjadi 2 bagian
                child[ship_name]["container"] = parent1[ship_name]["container"][:] # Jika kurang dari 0.5 maka child akan inherit dari parent1
            else:
                child[ship_name]["container"] = parent2[ship_name]["container"][:] # Jika lebih dari 0.5 maka child akan inherit dari parent2
        return child # Return child
    return parent1 # Jika probabiltas melebihi crossover rate maka parent akan lanjut ke generasi selanjutnya

# Fungsi mutation
def mutate(expedition): # Mutasi dilakukan dengan memindah container kapal ke kapal yang lain di dalam 1 populasi
    if random.random() < MUTATION_RATE:
        ship_from, ship_to = random.sample(list(expedition.keys()), 2) # Mengambil kapal dari 1 populasi

        if expedition[ship_from]["container"]:  # Cek apakah kapal memiliki muatan paling tidak 1
            container = random.choice(expedition[ship_from]["container"]) # Mengambil satu container acak

            total_weight_ship_to = sum(c["weight"] for c in expedition[ship_to]["container"]) # Menghitung total weight dari kapal tujuan
            if total_weight_ship_to + container["weight"] <= expedition[ship_to]["capacity"]: # Cek total weight dari kapal tujuan apakah melebihi capacity
                expedition[ship_from]["container"].remove(container) # Menghapus container dari list container kapal asal
                expedition[ship_to]["container"].append(container) # Menambah container dari kapal lain

    return expedition # Return 1 population

# Seleksi dengan Rank Selection
def rank_selection(population, fitnesses):
    ranked_population = sorted(zip(population, fitnesses), key=lambda x: x[1]) # Sort berdasarkan cost dari populasi
    
    ranked_population = [p for p in ranked_population if p[1] > 0] # Memastikan untuk tidak memilih populasi container kosong
    
    if not ranked_population: # Mencegah kebocoran ketika 1 populasi masih memiliki container yang kosong
        print("[ERROR] All expeditions are empty! Reinitializing...")
        return initialize_population()[0]

    ranks = list(range(len(ranked_population), 0, -1))  # List rank dengan biaya terendah
    total_ranks = sum(ranks) # Menghitung total rank
    selection_probs = [r / total_ranks for r in ranks] # Menghitung probabilitas per rank
    
    selected_index = np.random.choice(len(ranked_population), p=selection_probs) # Menyeleksi populasi berdasarkan rank
    return ranked_population[selected_index][0] # Return rank

# Debug display solusi (menampilkan generasi berapa dan populasi index berapa yang terpilih menjadi solusi)
def display_solution(solution, fitness, index=None):
    if index is not None:
        print(f"Solution #{index}: Cost = {format_rupiah(fitness)}")
    else:
        print(f"Solution: Cost = {format_rupiah(fitness)}")
        
    for ship_name, ship in solution.items():
        weight = sum(c["weight"] for c in ship["container"])
        containers_in_ship = len(ship["container"])
        if containers_in_ship > 0:
            print(f"  {ship_name}: {containers_in_ship} containers, Weight: {weight}/{ship['capacity']}")
            for container in ship["container"]:
                special = "Special" if container["isSpecial"] else "Regular"
                print(f"    - Container {container['id']} ({special}): Weight {container['weight']}, Cost {format_rupiah(container['cost'])}, To: {container['destination'][0]}")

# Debug display top 5 populasi untuk setiap generasi
def display_top_solutions(population, fitnesses, generation):
    print(f"\n=== Generation {generation}: Top 5 Solutions ===")
    sorted_solutions = sorted(zip(population, fitnesses), key=lambda x: x[1])
    
    for i, (solution, fitness) in enumerate(sorted_solutions[:5], 1):
        display_solution(solution, fitness, i)
        print()

# Algoritma Genetika
def genetic_algorithm():
    population = initialize_population()  # Inisialisasi populasi
    global TARGET_PRICE
    TARGET_PRICE = get_initial_target_price(population)  # Panggil get initial target price 

    best_solution = None # Inisialisasi best solution
    best_fitness = float("inf")  # Inisialisasi best fitness
    best_generation = -1 # Inisialisasi best generation
    best_population_index = -1 # Inisialisasi population index

    for gen in range(GEN_COUNT):
        fitnesses = [evaluate_fitness(ind) for ind in population] # Memilih parent berdasarkan fitness yang baik

        if (gen + 1) % 10 == 0 or gen == 0:  # Debug top 5 populasi dalam 1 generasi
            display_top_solutions(population, fitnesses, gen + 1)

        gen_best_idx = fitnesses.index(min(fitnesses)) # Melacak generasi
        gen_best_solution = population[gen_best_idx] # Melacak best populasi per generasi
        gen_best_fitness = fitnesses[gen_best_idx] # Melacak fitness terbaik per generasi

        assigned_containers = {c["id"] for s in gen_best_solution.values() for c in s["container"]} # Mengambil data container dari best populasi
        if gen_best_fitness < best_fitness and len(assigned_containers) == len(containers): # Cek apakah fitness per generasi lebih kecil dari best fitness dan apakah total assigned container sama dengan containers
            best_solution = gen_best_solution # Set best solution ke best solution per generasi
            best_fitness = gen_best_fitness # Set best fitness ke best fitness per generasi
            best_generation = gen + 1 # Set best generation ke generasi sekarang + 1
            best_population_index = gen_best_idx  # Set best population index menjadi generasi sekarang

        print(f"Generation {gen+1}: Best fitness = {format_rupiah(min(fitnesses))}, Avg fitness = {format_rupiah(sum(fitnesses)/len(fitnesses))}") # Debug best fitness dan avg fitness per generasi

        new_population = [] # inisialisasi populasi baru
        while len(new_population) < POP_SIZE:
            parent1, parent2 = rank_selection(population, fitnesses), rank_selection(population, fitnesses) # Memilih parent berdasarkan rank
            offspring = crossover(parent1, parent2) # Generate offspring
            new_population.append(mutate(offspring)) # Menambah populasi baru

        population = sorted(new_population, key=evaluate_fitness)[:POP_SIZE] # Sort berdasarkan fitness di dalam new population

    # Debug solusi terbaik
    print("\n=== FINAL BEST SOLUTION ===")
    print(f"✅ Picked from Generation {best_generation}, Population Index {best_population_index}")
    display_solution(best_solution, best_fitness)


if __name__ == "__main__":
    genetic_algorithm()