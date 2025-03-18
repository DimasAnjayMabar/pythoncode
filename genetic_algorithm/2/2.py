import random  # import library random untuk menghasilkan angka acak
import numpy as np  # import library numpy untuk operasi numerik
import pandas as pd  # import library pandas untuk manipulasi data
from tabulate import tabulate  # import library tabulate untuk menampilkan tabel
from copy import deepcopy  # import deepcopy untuk membuat salinan objek yang independen

# parameter algoritma genetika
POP_SIZE = 3  # ukuran populasi (jumlah kromosom dalam satu generasi)
GEN_COUNT = 3  # jumlah generasi yang akan dijalankan
CROSSOVER_RATE = 0.7  # tingkat crossover (probabilitas terjadinya crossover)
MUTATION_RATE = 0.05  # tingkat mutasi (probabilitas terjadinya mutasi)
SPECIAL_WEIGHT = 50000000  # biaya tambahan untuk container khusus (50 juta rupiah)

# fungsi untuk inisialisasi data kapal dengan kapasitas maksimal
def initialize_expedition():
    return {
        "Ship1": {"id": 1, "name": "Ship1", "capacity": 150, "container": []}, 
        "Ship2": {"id": 2, "name": "Ship2", "capacity": 160, "container": []},
        "Ship3": {"id": 3, "name": "Ship3", "capacity": 170, "container": []}  
    }

# data tujuan dengan jarak (semakin kecil, semakin dekat. jarak dalam km)
destinations = {
    1: {"id": 1, "destination": "Surabaya", "range": 760}, 
    2: {"id": 2, "destination": "Jakarta", "range": 400},   
    3: {"id": 3, "destination": "Medan", "range": 1900},   
    4: {"id": 4, "destination": "Manado", "range": 2100}, 
    5: {"id": 5, "destination": "Ambon", "range": 800},     
    6: {"id": 6, "destination": "Jayapura", "range": 2500},
    7: {"id": 7, "destination": "Batam", "range": 1300}, 
    8: {"id": 8, "destination": "Bali", "range": 950},    
    9: {"id": 9, "destination": "Makassar", "range": 1500},
    10: {"id": 10, "destination": "Pontianak", "range": 1000}, 
    11: {"id": 11, "destination": "Semarang", "range": 500}, 
    12: {"id": 12, "destination": "Banjarmasin", "range": 1200},
    13: {"id": 13, "destination": "Lombok", "range": 900}, 
    14: {"id": 14, "destination": "Padang", "range": 1700}, 
    15: {"id": 15, "destination": "Pekanbaru", "range": 1400}, 
    16: {"id": 16, "destination": "Balikpapan", "range": 1600}, 
    17: {"id": 17, "destination": "Jambi", "range": 1100}  
}

# data container dengan berat, status khusus, tujuan, dan biaya
containers = {
    1: {"id": 1, "weight": 30, "isSpecial": True, "destination": ["Surabaya", 760], "cost": 20000000 + SPECIAL_WEIGHT},  
    2: {"id": 2, "weight": 40, "isSpecial": False, "destination": ["Jakarta", 400], "cost": 30000000}, 
    3: {"id": 3, "weight": 50, "isSpecial": True, "destination": ["Medan", 1900], "cost": 40000000 + SPECIAL_WEIGHT}, 
    4: {"id": 4, "weight": 20, "isSpecial": False, "destination": ["Bali", 950], "cost": 18000000}, 
    5: {"id": 5, "weight": 45, "isSpecial": False, "destination": ["Makassar", 1500], "cost": 35000000},
    6: {"id": 6, "weight": 35, "isSpecial": True, "destination": ["Pontianak", 1000], "cost": 25000000 + SPECIAL_WEIGHT},  
    7: {"id": 7, "weight": 25, "isSpecial": False, "destination": ["Semarang", 500], "cost": 20000000}, 
    8: {"id": 8, "weight": 60, "isSpecial": True, "destination": ["Batam", 1300], "cost": 45000000 + SPECIAL_WEIGHT},  
    9: {"id": 9, "weight": 55, "isSpecial": False, "destination": ["Manado", 2100], "cost": 38000000},
    10: {"id": 10, "weight": 28, "isSpecial": True, "destination": ["Banjarmasin", 1200], "cost": 22000000 + SPECIAL_WEIGHT}, 
    11: {"id": 11, "weight": 38, "isSpecial": False, "destination": ["Lombok", 900], "cost": 27000000},  
    12: {"id": 12, "weight": 42, "isSpecial": True, "destination": ["Padang", 1700], "cost": 32000000 + SPECIAL_WEIGHT},
    13: {"id": 13, "weight": 50, "isSpecial": False, "destination": ["Pekanbaru", 1400], "cost": 34000000}, 
    14: {"id": 14, "weight": 48, "isSpecial": True, "destination": ["Balikpapan", 1600], "cost": 36000000 + SPECIAL_WEIGHT},  
    15: {"id": 15, "weight": 52, "isSpecial": False, "destination": ["Jambi", 1100], "cost": 31000000}, 
}

# fungsi untuk inisialisasi populasi awal
def initialize_population():
    population = []  # array untuk menyimpan ekspedisi (kromosom)
    for _ in range(POP_SIZE):  # membuat populasi sebanyak POP_SIZE
        expedition = initialize_expedition()  # inisialisasi satu ekspedisi (satu kromosom)
        container_list = list(containers.values())  # mengambil semua container
        random.shuffle(container_list)  # mengacak urutan container
        
        for container in container_list:
            possible_ships = [
                ship for ship in expedition.values() 
                    if sum(c["weight"] for c in ship["container"]) + container["weight"] <= ship["capacity"]]  # mencari kapal yang masih bisa menampung container
            if possible_ships:
                chosen_ship = random.choice(possible_ships)  # memilih kapal secara acak
                chosen_ship["container"].append(container)  # menambahkan container ke kapal yang dipilih
        
        population.append(expedition)  # menambahkan ekspedisi ke populasi
    return population

# fungsi untuk menghitung fitness (biaya total)
def evaluate_fitness(expedition):
    total_cost = sum(container["cost"] for ship in expedition.values() for container in ship["container"])  # menghitung total biaya semua container dalam ekspedisi
    
    # mengecek container yang tidak terangkut
    assigned_containers = {container["id"] for ship in expedition.values() for container in ship["container"]}
    total_containers = set(containers.keys())
    missing_containers = total_containers - assigned_containers
    
    if missing_containers:
        penalty = len(missing_containers) * 1_000_000_000  # penalti untuk container yang tidak terangkut
        total_cost += penalty
        print(f"[PENALTY] {len(missing_containers)} missing containers! adding {format_rupiah(penalty)} penalty.")
    
    return total_cost, missing_containers  # mengembalikan total biaya dan container yang tidak terangkut

# fungsi crossover dengan detail
def crossover(parent1, parent2, parent1_idx, parent2_idx):
    child = initialize_expedition()  # inisialisasi ekspedisi baru untuk anak

    crossover_details = {  # detail crossover
        "parent1_idx": parent1_idx,
        "parent2_idx": parent2_idx,
        "crossover_happened": False,
        "ship_inheritance": {}
    }
    
    if random.random() < CROSSOVER_RATE:  # jika terjadi crossover
        crossover_details["crossover_happened"] = True
        
        for ship_name in parent1.keys():  # untuk setiap kapal di parent1
            if random.random() < 0.5:  # memilih secara acak apakah akan mengambil container dari parent1 atau parent2
                child[ship_name]["container"] = parent1[ship_name]["container"][:]  # mengambil container dari parent1
                crossover_details["ship_inheritance"][ship_name] = "parent1"
            else:
                child[ship_name]["container"] = parent2[ship_name]["container"][:]  # mengambil container dari parent2
                crossover_details["ship_inheritance"][ship_name] = "parent2"
        
        return child, crossover_details
    
    # jika tidak terjadi crossover, kembalikan parent1
    for ship_name in parent1.keys():
        child[ship_name]["container"] = parent1[ship_name]["container"][:]
        crossover_details["ship_inheritance"][ship_name] = "parent1 (no crossover)"
    
    return child, crossover_details

# fungsi mutasi dengan detail
def mutate(expedition):
    expedition_copy = deepcopy(expedition)  # membuat salinan ekspedisi sebelum mutasi
    original_chromosome = solution_to_chromosome(expedition_copy)  # mengambil kromosom asli
    
    mutation_details = {
        "mutation_happened": False,
        "ship_from": None,
        "ship_to": None,
        "container_id": None,
        "original_chromosome": original_chromosome,
        "mutated_chromosome": None
    }
    
    if random.random() < MUTATION_RATE:  # jika terjadi mutasi
        ship_from, ship_to = random.sample(list(expedition.keys()), 2)  # memilih dua kapal secara acak
        mutation_details["ship_from"] = ship_from
        mutation_details["ship_to"] = ship_to

        if expedition[ship_from]["container"]:  # jika kapal asal memiliki container
            container = random.choice(expedition[ship_from]["container"])  # memilih container secara acak
            mutation_details["container_id"] = container["id"]

            # mengecek apakah kapal tujuan bisa menampung container
            total_weight_ship_to = sum(c["weight"] for c in expedition[ship_to]["container"])
            if total_weight_ship_to + container["weight"] <= expedition[ship_to]["capacity"]:
                expedition[ship_from]["container"].remove(container)  # menghapus container dari kapal asal
                expedition[ship_to]["container"].append(container)  # menambahkan container ke kapal tujuan
                mutation_details["mutation_happened"] = True
    
    # mengambil kromosom setelah mutasi
    mutation_details["mutated_chromosome"] = solution_to_chromosome(expedition)
    
    return expedition, mutation_details

# fungsi seleksi dengan rank selection
def rank_selection(population, fitnesses, selection_index):
    # mengurutkan populasi berdasarkan fitness (biaya terendah adalah yang terbaik)
    ranked_population = sorted(zip(range(len(population)), population, fitnesses), key=lambda x: x[2])
    
    # menyaring solusi yang valid
    ranked_population = [(idx, p, f) for idx, p, f in ranked_population if f > 0]
    
    if not ranked_population:  # jika semua solusi tidak valid
        print("[error] all expeditions are empty! reinitializing...")
        new_population = initialize_population()[0]
        return new_population, {
            "selection_id": selection_index,
            "selection_method": "rank selection (error - reinitialized)",
            "population_size": len(population),
            "ranks": [],
            "probabilities": [],
            "selected_idx": -1
        }

    # menghitung rank dan probabilitas seleksi
    idx_list = [idx for idx, _, _ in ranked_population]
    ranks = list(range(len(ranked_population), 0, -1))  # rank tertinggi adalah biaya terendah
    total_ranks = sum(ranks)
    selection_probs = [r / total_ranks for r in ranks]
    
    # memilih solusi berdasarkan probabilitas
    selected_index = np.random.choice(len(ranked_population), p=selection_probs)
    original_idx, selected_solution, selected_fitness = ranked_population[selected_index]
    
    # detail seleksi
    selection_details = {
        "selection_id": selection_index,
        "selection_method": "rank selection",
        "population_size": len(population),
        "ranks": list(zip(idx_list, ranks)),
        "probabilities": list(zip(idx_list, selection_probs)),
        "selected_idx": original_idx,
        "selected_fitness": selected_fitness
    }
    
    return selected_solution, selection_details

# algoritma genetika utama
def genetic_algorithm():
    # inisialisasi populasi awal
    population = initialize_population()
    global target_price
    target_price = get_initial_target_price(population)

    best_solution = None  # solusi terbaik
    best_fitness = float("inf")  # fitness terbaik (biaya terendah)
    best_missing_containers = set()  # container yang tidak terangkut pada solusi terbaik

    for gen in range(GEN_COUNT):  # iterasi sebanyak gen_count generasi
        print(f"\n{'='*80}")
        print(f"generation {gen+1}")
        print(f"{'='*80}")
        
        # evaluasi fitness setiap individu dalam populasi
        fitnesses = []
        missing_containers_list = []
        for ind in population:
            fitness, missing_containers = evaluate_fitness(ind)
            fitnesses.append(fitness)
            missing_containers_list.append(missing_containers)
        
        # menampilkan tabel kromosom
        display_chromosome_table(population, fitnesses, f"generation {gen+1}: raw solutions/chromosomes")
        
        # mencari solusi terbaik dalam generasi ini
        gen_best_idx = fitnesses.index(min(fitnesses))
        gen_best_solution = population[gen_best_idx]
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_missing_containers = missing_containers_list[gen_best_idx]
        
        print(f"\n* generation {gen+1} best expedition (index {gen_best_idx}):")
        display_solution(gen_best_solution, gen_best_fitness, gen_best_missing_containers, gen_best_idx)
        
        # memperbarui solusi terbaik secara keseluruhan
        if gen_best_fitness < best_fitness:
            best_solution = gen_best_solution
            best_fitness = gen_best_fitness
            best_missing_containers = gen_best_missing_containers
        
        print(f"\ngeneration {gen+1} stats:")
        print(f"- best fitness: {format_rupiah(min(fitnesses))}")
        print(f"- average fitness: {format_rupiah(sum(fitnesses)/len(fitnesses))}")
        print(f"- current best fitness overall: {format_rupiah(best_fitness)}")
        
        # membuat populasi baru
        new_population = []
        crossover_details_list = []
        mutation_details_list = []
        
        # loop untuk seleksi dan crossover
        while len(new_population) < POP_SIZE:
            # memilih parent
            parent1, selection1 = rank_selection(population, fitnesses, len(new_population))
            parent2, selection2 = rank_selection(population, fitnesses, len(new_population) + 1)
            
            # menghasilkan offspring
            offspring, crossover_details = crossover(parent1, parent2, selection1["selected_idx"], selection2["selected_idx"])
            crossover_details_list.append(crossover_details)
            
            # melakukan mutasi
            mutated_offspring, mutation_details = mutate(offspring)
            mutation_details_list.append(mutation_details)
            
            # menambahkan offspring ke populasi baru
            new_population.append(mutated_offspring)
        
        # menggantikan populasi lama dengan populasi baru
        population = new_population
    
    # menampilkan solusi terbaik akhir
    print("\n" + "="*80)
    print("final best solution")
    print("="*80)
    display_solution(best_solution, best_fitness, best_missing_containers)
    
    # menampilkan kromosom solusi terbaik
    best_chromosome = solution_to_chromosome(best_solution)
    print(f"\nbest solution chromosome: {' '.join(map(str, best_chromosome))}")
    print(f"final fitness: {format_rupiah(best_fitness)}")

# DEBUG FUNCTIONS

# debug mutation details
def display_mutations(mutation_details_list, generation):
    print(f"\n=== Generation {generation}: Mutation Details ===")
    
    mutation_count = sum(1 for m in mutation_details_list if m["mutation_happened"])
    mutation_percentage = (mutation_count / len(mutation_details_list)) * 100
    print(f"Mutations applied: {mutation_count}/{len(mutation_details_list)} ({mutation_percentage:.1f}%)")
    
    mutation_data = []
    for i, details in enumerate(mutation_details_list):
        if details["mutation_happened"]:
            mutation_data.append({
                'Expedition': i,
                'Container': details["container_id"],
                'From Ship': details["ship_from"],
                'To Ship': details["ship_to"],
                'Before': ' '.join(map(str, details["original_chromosome"])),
                'After': ' '.join(map(str, details["mutated_chromosome"]))
            })
    
    if mutation_data:
        df = pd.DataFrame(mutation_data)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    else:
        print("No mutations occurred in this generation.")

# debug crossover details
def display_crossovers(crossover_details_list, generation):
    print(f"\n=== Generation {generation}: Crossover Details ===")
    
    crossover_data = []
    for i, details in enumerate(crossover_details_list):
        inheritance = ", ".join([f"{ship}: {parent}" for ship, parent in details["ship_inheritance"].items()])
        crossover_data.append({
            'Offspring': i,
            'Parent 1': details["parent1_idx"],
            'Parent 2': details["parent2_idx"],
            'Crossover Applied': details["crossover_happened"],
            'Inheritance': inheritance
        })
    
    df = pd.DataFrame(crossover_data)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

# debug selection detail
def display_solution(solution, fitness, missing_containers, index=None):
    if index is not None:
        print(f"Expedition #{index}: Cost = {format_rupiah(fitness)}")
    else:
        print(f"Expedition: Cost = {format_rupiah(fitness)}")
        
    for ship_name, ship in solution.items():
        weight = sum(c["weight"] for c in ship["container"])
        containers_in_ship = len(ship["container"])
        if containers_in_ship > 0:
            print(f"  {ship_name}: {containers_in_ship} containers, Weight: {weight}/{ship['capacity']}")
            for container in ship["container"]:
                special = "Special" if container["isSpecial"] else "Regular"
                print(f"    - Container {container['id']} ({special}): Weight {container['weight']}, Cost {format_rupiah(container['cost'])}, To: {container['destination'][0]}")
    
    # display container hilang
    if missing_containers:
        print("\n  Missing Containers:")
        for container_id in missing_containers:
            container = containers[container_id]
            special = "Special" if container["isSpecial"] else "Regular"
            print(f"    - Container {container_id} ({special}): Weight {container['weight']}, Cost {format_rupiah(container['cost'])}, To: {container['destination'][0]}")
    else:
        print("\n  All containers assigned.")

# debug detail ekspedisi
def display_solution(solution, fitness, index=None):
    if index is not None:
        print(f"Expedition #{index}: Cost = {format_rupiah(fitness)}")
    else:
        print(f"Expedition: Cost = {format_rupiah(fitness)}")
        
    for ship_name, ship in solution.items():
        weight = sum(c["weight"] for c in ship["container"])
        containers_in_ship = len(ship["container"])
        if containers_in_ship > 0:
            print(f"  {ship_name}: {containers_in_ship} containers, Weight: {weight}/{ship['capacity']}")
            for container in ship["container"]:
                special = "Special" if container["isSpecial"] else "Regular"
                print(f"    - Container {container['id']} ({special}): Weight {container['weight']}, Cost {format_rupiah(container['cost'])}, To: {container['destination'][0]}")

# Tampilkan tabel kromosom
def display_chromosome_table(population, fitnesses, title):
    """Display chromosomes in a table format"""
    print(f"\n=== {title} ===")
    print("Legend: 0=Not assigned, 1=Ship1, 2=Ship2, 3=Ship3")
    print("Each position represents a container, e.g., '1 2 3' means Container1→Ship1, Container2→Ship2, Container3→Ship3")
    
    chromosome_data = []
    for i, (solution, fitness) in enumerate(zip(population, fitnesses)):
        chromosome = solution_to_chromosome(solution)
        missing = chromosome.count(0)
        # hitung container per kapal
        ship_counts = {
            "Ship1": chromosome.count(1),
            "Ship2": chromosome.count(2),
            "Ship3": chromosome.count(3)
        }
        
        chromosome_data.append({
            'Expedition': i,
            'Container Assigned to Which Ship': ' '.join(map(str, chromosome)),
            'Ship1': ship_counts["Ship1"],
            'Ship2': ship_counts["Ship2"],
            'Ship3': ship_counts["Ship3"],
            'Missing': missing,
            'Fitness': format_rupiah(fitness)
        })
    
    df = pd.DataFrame(chromosome_data)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


# visualisasi kromosom (representasi solusi)
def solution_to_chromosome(solution):
    """
    Convert expedition solution to chromosome representation
    Each position in the chromosome represents a container
    The value represents which ship the container is assigned to:
    1 = Ship1, 2 = Ship2, 3 = Ship3, 0 = not assigned
    """
    chromosome = []
    for container_id in range(1, len(containers) + 1):
        found = False
        for ship_name, ship in solution.items():
            for container in ship["container"]:
                if container["id"] == container_id:
                    chromosome.append(int(ship_name[-1]))  # Ship1 -> 1, Ship2 -> 2, Ship3 -> 3
                    found = True
                    break
            if found:
                break
        if not found:
            chromosome.append(0)  # 0 menandakan tidak di assign di kapal manapun
    return chromosome

#MISC FUNCTIONS 

# Fungsi inisialisasi target price yang dynamic agar meningkatkan kinerja fitness function
def get_initial_target_price(population):
    avg_cost = sum(evaluate_fitness(ind)[0] for ind in population) / len(population)  # Menghitung average cost dari initial population
    return avg_cost * 1.2  # Set 20% lebih tinggi dari avg cost untuk target price

# Konversi ke format mata uang Rupiah
def format_rupiah(amount):
    return f"Rp {amount:,.0f}".replace(",", ".")

if __name__ == "__main__":
    genetic_algorithm()