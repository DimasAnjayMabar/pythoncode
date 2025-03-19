import random
import numpy as np
import pandas as pd
from tabulate import tabulate
from copy import deepcopy

# Parameter algoritma genetika
POP_SIZE = 50
GEN_COUNT = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
SPECIAL_WEIGHT = 1000
COST_PER_WEIGHT = 2500


# Fungsi untuk inisialisasi data kapal dengan kapasitas maksimal
def initialize_expedition():
    return {
        "Ship1": {"id": 1, "name": "Ship1", "capacity": 150, "container": [], "fuel_tank": 20000}, 
        "Ship2": {"id": 2, "name": "Ship2", "capacity": 160, "container": [], "fuel_tank": 20000},
        "Ship3": {"id": 3, "name": "Ship3", "capacity": 170, "container": [], "fuel_tank": 20000}  
    }

# Data tujuan dengan jarak (semakin kecil, semakin dekat. jarak dalam km)
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

# Data container dengan berat, status khusus, tujuan, dan biaya
containers = {
    1: {"id": 1, "weight": 30, "isSpecial": True, "destination": ["Surabaya", 760], "cost": None},  
    2: {"id": 2, "weight": 40, "isSpecial": False, "destination": ["Jakarta", 400], "cost": None}, 
    3: {"id": 3, "weight": 50, "isSpecial": True, "destination": ["Medan", 1900], "cost": None}, 
    4: {"id": 4, "weight": 20, "isSpecial": False, "destination": ["Bali", 950], "cost": None}, 
    5: {"id": 5, "weight": 45, "isSpecial": False, "destination": ["Makassar", 1500], "cost": None},
    6: {"id": 6, "weight": 35, "isSpecial": True, "destination": ["Pontianak", 1000], "cost": None},  
    7: {"id": 7, "weight": 25, "isSpecial": False, "destination": ["Semarang", 500], "cost": None}, 
    8: {"id": 8, "weight": 60, "isSpecial": True, "destination": ["Batam", 1300], "cost": None},  
    9: {"id": 9, "weight": 55, "isSpecial": False, "destination": ["Manado", 2100], "cost": None},
    10: {"id": 10, "weight": 28, "isSpecial": True, "destination": ["Banjarmasin", 1200], "cost": None}, 
    11: {"id": 11, "weight": 38, "isSpecial": False, "destination": ["Lombok", 900], "cost": None},  
    12: {"id": 12, "weight": 42, "isSpecial": True, "destination": ["Padang", 1700], "cost": None},
    13: {"id": 13, "weight": 50, "isSpecial": False, "destination": ["Pekanbaru", 1400], "cost": None}, 
    14: {"id": 14, "weight": 48, "isSpecial": True, "destination": ["Balikpapan", 1600], "cost": None},  
    15: {"id": 15, "weight": 52, "isSpecial": False, "destination": ["Jambi", 1100], "cost": None}, 
}

# Fungsi untuk inisialisasi populasi awal
def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        expedition = initialize_expedition()
        container_list = list(containers.values())
        # sort container berdasarkan jarak terdekat
        container_list.sort(key=lambda x: x["destination"][1])
        
        for container in container_list:
            possible_ships = [
                ship for ship in expedition.values() 
                    if sum(c["weight"] for c in ship["container"]) + container["weight"] <= ship["capacity"]]
            if possible_ships:
                chosen_ship = random.choice(possible_ships)
                chosen_ship["container"].append(container)
        
        population.append(expedition)
    return population

# Fungsi untuk menghitung fitness
def evaluate_fitness(expedition):
    total_cost = 0
    FUEL_EFFICIENCY = 5  # 1 km = 5 liter bensin

    for ship in expedition.values():
        if ship["container"]:
            total_weight = sum(c["weight"] for c in ship["container"])
            max_distance = max(c["destination"][1] for c in ship["container"])  # jarak terjauh 
            
            # bensin yang dibutuhkan kapal
            fuel_needed = max_distance * FUEL_EFFICIENCY  # dalam liter
            
            # check apakah kapal bisa mencapai semua destinasi dengan bensin yang ada
            if max_distance > ship["fuel_tank"]:
                total_cost += 1e7  # penalti ketika kapal tidak sampai (hanya untuk mengeliminasi solusi dalam GA, bukan representasi biaya asli)
            else:
                # cost berdasarkan berat dan jaraknya (asumsi semua container volumenya sama)
                ship_cost = total_weight * max_distance * COST_PER_WEIGHT
                
                # fuel cost
                fuel_cost = fuel_needed * 1000  # asumsi 1000 dollar per liter 

                # biaya untuk special container
                special_penalty = sum(SPECIAL_WEIGHT for c in ship["container"] if c["isSpecial"])
                
                # total cost 1 kapal
                total_cost += ship_cost + special_penalty + fuel_cost

    return total_cost

# Fungsi crossover dengan detail (single point crossover)
def crossover(parent1, parent2, parent1_idx, parent2_idx):
    child = initialize_expedition()  # inisialisasi offspring
    crossover_details = {
        "parent1_idx": parent1_idx,
        "parent2_idx": parent2_idx,
        "crossover_happened": False,
        "crossover_point": None,
        "ship_inheritance": {}
    }

    if random.random() < CROSSOVER_RATE:  # check posibility dari semua populasi di satu generasi
        crossover_details["crossover_happened"] = True

        # inisialisasi parent
        parent1_chromosome = solution_to_chromosome(parent1)
        parent2_chromosome = solution_to_chromosome(parent2)

        # memilih crossover point
        crossover_point = random.randint(1, len(parent1_chromosome) - 1)
        crossover_details["crossover_point"] = crossover_point

        # single point crossover
        child_chromosome = parent1_chromosome[:crossover_point] + parent2_chromosome[crossover_point:]

        # convert kembali child ke mode expedition
        child = chromosome_to_solution(child_chromosome)

        # melacak inheritance
        for ship_name in child.keys():
            crossover_details["ship_inheritance"][ship_name] = "parent1" if crossover_point < len(parent1_chromosome) // 2 else "parent2"

        return child, crossover_details

    # jika tidak ada crossover return parent
    for ship_name in parent1.keys():
        child[ship_name]["container"] = parent1[ship_name]["container"][:]
        crossover_details["ship_inheritance"][ship_name] = "parent1 (no crossover)"

    return child, crossover_details

# Fungsi mutasi dengan detail
def mutate(expedition):
    expedition_copy = deepcopy(expedition)
    original_chromosome = solution_to_chromosome(expedition_copy)
    mutation_details = {
        "mutation_happened": False,
        "ship_from": None,
        "ship_to": None,
        "container_id": None,
        "original_chromosome": original_chromosome,
        "mutated_chromosome": None
    }
    
    if random.random() < MUTATION_RATE:
        ship_from, ship_to = random.sample(list(expedition.keys()), 2)
        mutation_details["ship_from"] = ship_from
        mutation_details["ship_to"] = ship_to

        if expedition[ship_from]["container"]:
            container = random.choice(expedition[ship_from]["container"])
            mutation_details["container_id"] = container["id"]
            
            total_weight_ship_to = sum(c["weight"] for c in expedition[ship_to]["container"])
            if total_weight_ship_to + container["weight"] <= expedition[ship_to]["capacity"]:
                # memindahkan container ke kapal lain   
                expedition[ship_from]["container"].remove(container)
                expedition[ship_to]["container"].append(container)
                mutation_details["mutation_happened"] = True

    
    mutation_details["mutated_chromosome"] = solution_to_chromosome(expedition)
    return expedition, mutation_details

# Fungsi seleksi dengan roulette wheel selection
def roulette_wheel_selection(population, fitnesses, selection_index):
    fitnesses = np.array(fitnesses)
    # membalik fitness karena cost terendah berarti lebih bagus
    inverted_fitnesses = 1 / (fitnesses + 1e-10)
    total_fitness = np.sum(inverted_fitnesses)
    selection_probs = inverted_fitnesses / total_fitness
    
    selected_index = np.random.choice(len(population), p=selection_probs)
    selected_solution = population[selected_index]
    
    selection_details = {
        "selection_id": selection_index,
        "selection_method": "roulette wheel selection",
        "population_size": len(population),
        "probabilities": list(zip(range(len(population)), selection_probs)),
        "selected_idx": selected_index,
        "selected_fitness": fitnesses[selected_index]
    }
    
    return selected_solution, selection_details

# Algoritma genetika utama
def genetic_algorithm():
    population = initialize_population()
    global_best_solution = None
    global_best_fitness = float("inf")
    global_best_generation = -1
    global_best_index = -1

    for gen in range(GEN_COUNT):
        print(f"\n{'='*80}")
        print(f"GENERATION {gen+1}")
        print(f"{'='*80}")
        
        fitnesses = [evaluate_fitness(ind) for ind in population]
        display_chromosome_table(population, fitnesses, f"Generation {gen+1}: Raw Solutions/Chromosomes")
        
        gen_best_idx = fitnesses.index(min(fitnesses))
        gen_best_solution = population[gen_best_idx]
        gen_best_fitness = fitnesses[gen_best_idx]
        
        if gen_best_fitness < global_best_fitness:
            global_best_solution = gen_best_solution
            global_best_fitness = gen_best_fitness
            global_best_generation = gen + 1
            global_best_index = gen_best_idx
        
        print(f"\n* Generation {gen+1} Best Expedition (Index {gen_best_idx}):")
        display_solution(gen_best_solution, gen_best_fitness, gen_best_idx)
        
        print(f"\nGeneration {gen+1} Stats:")
        print(f"- Best fitness: {min(fitnesses)}")
        print(f"- Average fitness: {sum(fitnesses)/len(fitnesses)}")
        print(f"- Current best fitness overall: {global_best_fitness}")
        
        new_population = []
        crossover_details_list = []
        mutation_details_list = []
        
        while len(new_population) < POP_SIZE:
            parent1, selection1 = roulette_wheel_selection(population, fitnesses, len(new_population))
            parent2, selection2 = roulette_wheel_selection(population, fitnesses, len(new_population) + 1)
            offspring, crossover_details = crossover(parent1, parent2, selection1["selected_idx"], selection2["selected_idx"])
            crossover_details_list.append(crossover_details)
            mutated_offspring, mutation_details = mutate(offspring)
            mutation_details_list.append(mutation_details)
            new_population.append(mutated_offspring)
        
        population = new_population
    
    print("\n" + "="*80)
    print("FINAL BEST SOLUTION")
    print("="*80)
    print(f"Best solution found in Generation {global_best_generation}, Population Index {global_best_index}:")
    display_solution(global_best_solution, global_best_fitness, global_best_index)
    print("\nBest Solution Chromosome (Table View):")
    display_chromosome_table([global_best_solution], [global_best_fitness], "Best Solution Chromosome")
    best_chromosome = solution_to_chromosome(global_best_solution)

# DEBUG FUNCTIONS

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
            'Crossover Point': details["crossover_point"] if details["crossover_happened"] else "N/A",
            'Inheritance': inheritance
        })
    df = pd.DataFrame(crossover_data)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

def display_solution(solution, fitness, index=None):
    if index is not None:
        print(f"Expedition #{index}: Total Cost = {format_rupiah(fitness)}")
    else:
        print(f"Expedition: Total Cost = {format_rupiah(fitness)}")
    
    FUEL_EFFICIENCY = 5  # 5 liters per km

    for ship_name, ship in solution.items():
        weight = sum(c["weight"] for c in ship["container"])
        containers_in_ship = len(ship["container"])
        if containers_in_ship > 0:
            max_distance = max(c["destination"][1] for c in ship["container"])
            fuel_needed = max_distance * FUEL_EFFICIENCY
            print(f"  {ship_name}: {containers_in_ship} containers, Weight: {weight}/{ship['capacity']}")
            print(f"    Estimated Fuel Needed: {fuel_needed:.2f} liters / Tank Range: {ship['fuel_tank']} km")
            sorted_containers = sorted(ship["container"], key=lambda c: c["destination"][1])
            for container in sorted_containers:
                special = "Special" if container["isSpecial"] else "Regular"
                cost_display = format_rupiah(container["cost"])
                print(f"    - Container {container['id']} ({special}): Weight {container['weight']}, "
                      f"To: {container['destination'][0]}, Distance: {container['destination'][1]}, Cost: {cost_display}")

def display_chromosome_table(population, fitnesses, title):
    print(f"\n=== {title} ===")
    print("Legend: Each row represents an expedition. Containers are grouped by the ship they are assigned to.")
    table_data = []
    for idx, (expedition, fitness) in enumerate(zip(population, fitnesses)):
        ship_assignments = {f"Ship{i}": [] for i in range(1, 4)}
        for ship_name, ship in expedition.items():
            for container in ship["container"]:
                ship_assignments[ship_name].append(container["id"])
        for ship_name in ship_assignments:
            ship_assignments[ship_name] = ", ".join(map(str, sorted(ship_assignments[ship_name]))) if ship_assignments[ship_name] else "None"
        
        # ✅ Format the fitness as Rupiah here:
        table_data.append([
            idx,
            ship_assignments["Ship1"],
            ship_assignments["Ship2"],
            ship_assignments["Ship3"],
            format_rupiah(fitness)  # ✅ Add formatter HERE
        ])

    headers = ["Expedition", "Ship1 Containers", "Ship2 Containers", "Ship3 Containers", "Fitness (Cost)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def solution_to_chromosome(solution):
    """
    Convert an expedition (solution) to a chromosome representation.
    Each position in the chromosome represents a container.
    The value represents which ship the container is assigned to:
    1 = Ship1, 2 = Ship2, 3 = Ship3, 0 = not assigned.
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
            chromosome.append(0)  # 0 means the container is not assigned to any ship
    return chromosome

def chromosome_to_solution(chromosome):
    """
    Convert a chromosome back to an expedition (solution).
    """
    expedition = initialize_expedition()  # Initialize a new expedition
    for container_id, ship_assignment in enumerate(chromosome, start=1):
        if ship_assignment != 0:  # If the container is assigned to a ship
            ship_name = f"Ship{ship_assignment}"
            container = containers[container_id]
            expedition[ship_name]["container"].append(container)
    return expedition

# MISC FUNCTIONS 

def format_rupiah(amount):
    return f"$ {amount:,.0f}".replace(",", ".")

def count_cost(weight, is_special):
    total_cost = weight * COST_PER_WEIGHT
    if is_special:
        total_cost += SPECIAL_WEIGHT
    return total_cost

for container_id, container_info in containers.items():
    weight = container_info["weight"]
    is_special = container_info["isSpecial"]
    container_info["cost"] = count_cost(weight, is_special)

if __name__ == "__main__":
    genetic_algorithm()