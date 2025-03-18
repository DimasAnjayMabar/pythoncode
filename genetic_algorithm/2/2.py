import random
import numpy as np
import pandas as pd
from tabulate import tabulate
from copy import deepcopy

# Parameter GA
POP_SIZE = 3  # Number of chromosomes (expeditions) in the population
GEN_COUNT = 3  # Number of generations
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05
SPECIAL_WEIGHT = 50000000  # 50jt Rupiah

# Konversi ke format mata uang Rupiah
def format_rupiah(amount):
    return f"Rp {amount:,.0f}".replace(",", ".")

# Data kapal dengan kapasitas maksimal
def initialize_expedition():
    return {
        "Ship1": {"id": 1, "name": "Ship1", "capacity": 150, "container": []},
        "Ship2": {"id": 2, "name": "Ship2", "capacity": 160, "container": []},
        "Ship3": {"id": 3, "name": "Ship3", "capacity": 170, "container": []}
    }

# Data tujuan dengan jarak (semakin kecil, semakin dekat. Jarak dalam km)
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

# Data peti kemas (berat dalam ton, cost dalam Rp)
SPECIAL_WEIGHT = 5000000  # Tambahan biaya jika container khusus

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

# Fungsi inisialisasi target price yang dynamic agar meningkatkan kinerja fitness function
def get_initial_target_price(population):
    avg_cost = sum(evaluate_fitness(ind) for ind in population) / len(population) # Menghitung average cost dari initial population
    return avg_cost * 1.2 # Set 20% lebih tinggi dari avg cost untuk target price

# Visualisasi kromosom (representasi solusi)
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
            chromosome.append(0)  # 0 means not assigned
    return chromosome

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
        # Count containers per ship
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

# Display detailed expedition
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

# Inisialisasi populasi
def initialize_population():
    population = []  # Array of expeditions (chromosomes)
    for _ in range(POP_SIZE):  # Create POP_SIZE expeditions
        expedition = initialize_expedition()  # 1 expedition = 1 chromosome (solution with 3 ships)
        container_list = list(containers.values())  # Get all containers
        random.shuffle(container_list)  # Shuffle containers for random assignment
        
        for container in container_list:
            possible_ships = [
                ship for ship in expedition.values() 
                    if sum(c["weight"] for c in ship["container"]) + container["weight"] <= ship["capacity"]]  # Ships with enough capacity
            if possible_ships:
                chosen_ship = random.choice(possible_ships)  # Choose a random suitable ship
                chosen_ship["container"].append(container)  # Add container to the ship
        
        population.append(expedition)  # Add expedition to population
    return population

# Fungsi fitness
def evaluate_fitness(expedition):
    total_cost = sum(container["cost"] for ship in expedition.values() for container in ship["container"])  # Total cost of all containers
    
    # Penalti untuk container yang tidak terangkut
    assigned_containers = {container["id"] for ship in expedition.values() for container in ship["container"]}
    total_containers = set(containers.keys())
    missing_containers = total_containers - assigned_containers
    
    if missing_containers:
        penalty = len(missing_containers) * 1_000_000_000  # Huge penalty for missing containers
        total_cost += penalty
        print(f"[PENALTY] {len(missing_containers)} missing containers! Adding {format_rupiah(penalty)} penalty.")

    return total_cost

# Fungsi crossover dengan detail
def crossover(parent1, parent2, parent1_idx, parent2_idx):
    child = initialize_expedition()  # Create a new empty expedition
    
    # For tracking crossover details
    crossover_details = {
        "parent1_idx": parent1_idx,
        "parent2_idx": parent2_idx,
        "crossover_happened": False,
        "ship_inheritance": {}
    }
    
    if random.random() < CROSSOVER_RATE:  # Apply crossover if random number is less than CROSSOVER_RATE
        crossover_details["crossover_happened"] = True
        
        for ship_name in parent1.keys():  # For each ship (Ship1, Ship2, Ship3)
            if random.random() < 0.5:  # 50% chance to inherit from each parent
                child[ship_name]["container"] = parent1[ship_name]["container"][:]  # Copy containers from parent1
                crossover_details["ship_inheritance"][ship_name] = "parent1"
            else:
                child[ship_name]["container"] = parent2[ship_name]["container"][:]  # Copy containers from parent2
                crossover_details["ship_inheritance"][ship_name] = "parent2"
        
        # Return child and crossover details
        return child, crossover_details
    
    # If no crossover, child is identical to parent1
    for ship_name in parent1.keys():
        child[ship_name]["container"] = parent1[ship_name]["container"][:]
        crossover_details["ship_inheritance"][ship_name] = "parent1 (no crossover)"
    
    return child, crossover_details

# Fungsi mutation dengan detail
def mutate(expedition):
    expedition_copy = deepcopy(expedition)  # Create a copy to compare before/after
    original_chromosome = solution_to_chromosome(expedition_copy)
    
    mutation_details = {
        "mutation_happened": False,
        "ship_from": None,
        "ship_to": None,
        "container_id": None,
        "original_chromosome": original_chromosome,
        "mutated_chromosome": None
    }
    
    if random.random() < MUTATION_RATE:  # Apply mutation if random number is less than MUTATION_RATE
        ship_from, ship_to = random.sample(list(expedition.keys()), 2)  # Select two random ships
        mutation_details["ship_from"] = ship_from
        mutation_details["ship_to"] = ship_to

        if expedition[ship_from]["container"]:  # Check if source ship has containers
            container = random.choice(expedition[ship_from]["container"])  # Select a random container
            mutation_details["container_id"] = container["id"]

            # Check if destination ship has enough capacity
            total_weight_ship_to = sum(c["weight"] for c in expedition[ship_to]["container"])
            if total_weight_ship_to + container["weight"] <= expedition[ship_to]["capacity"]:
                expedition[ship_from]["container"].remove(container)  # Remove from source ship
                expedition[ship_to]["container"].append(container)  # Add to destination ship
                mutation_details["mutation_happened"] = True
    
    # Capture the chromosome after mutation
    mutation_details["mutated_chromosome"] = solution_to_chromosome(expedition)
    
    return expedition, mutation_details

# Seleksi dengan Rank Selection (dengan detail proses)
def rank_selection(population, fitnesses, selection_index):
    # Sort by fitness (lower cost is better)
    ranked_population = sorted(zip(range(len(population)), population, fitnesses), key=lambda x: x[2])
    
    # Filter out invalid solutions
    ranked_population = [(idx, p, f) for idx, p, f in ranked_population if f > 0]
    
    if not ranked_population:  # Handle case where all solutions are invalid
        print("[ERROR] All expeditions are empty! Reinitializing...")
        new_population = initialize_population()[0]
        return new_population, {
            "selection_id": selection_index,
            "selection_method": "Rank Selection (Error - Reinitialized)",
            "population_size": len(population),
            "ranks": [],
            "probabilities": [],
            "selected_idx": -1
        }

    # Calculate ranks and probabilities
    idx_list = [idx for idx, _, _ in ranked_population]
    ranks = list(range(len(ranked_population), 0, -1))  # Highest rank for lowest cost
    total_ranks = sum(ranks)
    selection_probs = [r / total_ranks for r in ranks]
    
    # Select based on probabilities
    selected_index = np.random.choice(len(ranked_population), p=selection_probs)
    original_idx, selected_solution, selected_fitness = ranked_population[selected_index]
    
    # Prepare selection details
    selection_details = {
        "selection_id": selection_index,
        "selection_method": "Rank Selection",
        "population_size": len(population),
        "ranks": list(zip(idx_list, ranks)),
        "probabilities": list(zip(idx_list, selection_probs)),
        "selected_idx": original_idx,
        "selected_fitness": selected_fitness
    }
    
    return selected_solution, selection_details

# Display selection details
def display_selections(selection_details_list, generation):
    print(f"\n=== Generation {generation}: Selection Details ===")
    
    selection_data = []
    for details in selection_details_list:
        selection_data.append({
            'Selection ID': details["selection_id"],
            'Selected Expedition': details["selected_idx"],
            'Fitness': format_rupiah(details["selected_fitness"])
        })
    
    df = pd.DataFrame(selection_data)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Display rank probabilities for the first selection as an example
    if selection_details_list:
        first_selection = selection_details_list[0]
        print("\nRank Selection Probabilities Example (from first selection):")
        rank_data = []
        for idx, prob in first_selection["probabilities"]:
            rank_data.append({
                'Expedition': idx,
                'Probability': f"{prob:.4f}"
            })
        df_ranks = pd.DataFrame(rank_data)
        print(tabulate(df_ranks, headers='keys', tablefmt='grid', showindex=False))

# Display crossover details
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

# Display mutation details
def display_mutations(mutation_details_list, generation):
    print(f"\n=== Generation {generation}: Mutation Details ===")
    
    # Count mutations
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

# Algoritma Genetika
def genetic_algorithm():
    # Display explanatory text about the genetic algorithm structure
    print("\n" + "="*80)
    print("GENETIC ALGORITHM FOR EXPEDITION PLANNING")
    print("="*80)
    print("Structure:")
    print("- Population: Collection of multiple expedition solutions")
    print("- Expedition: One solution (chromosome) containing 3 ships")
    print("- Chromosome Representation: Shows which ship each container is assigned to")
    print("  Position = Container ID, Value = Ship (1=Ship1, 2=Ship2, 3=Ship3, 0=Not assigned)")
    print("- Fitness: Total cost of the expedition (lower is better)")
    print("="*80)
    
    # Initialize population
    population = initialize_population()
    global TARGET_PRICE
    TARGET_PRICE = get_initial_target_price(population)

    best_solution = None
    best_fitness = float("inf")
    best_generation = -1
    best_population_index = -1

    for gen in range(GEN_COUNT):
        print(f"\n{'='*80}")
        print(f"GENERATION {gen+1}")
        print(f"{'='*80}")
        
        # Evaluate population
        fitnesses = [evaluate_fitness(ind) for ind in population]
        
        # Display population chromosomes
        display_chromosome_table(population, fitnesses, f"Generation {gen+1}: Raw Solutions/Chromosomes")
        
        # Track best solution in generation
        gen_best_idx = fitnesses.index(min(fitnesses))
        gen_best_solution = population[gen_best_idx]
        gen_best_fitness = fitnesses[gen_best_idx]
        
        print(f"\n* Generation {gen+1} Best Expedition (Index {gen_best_idx}):")
        display_solution(gen_best_solution, gen_best_fitness)
        
        # Track all time best solution
        assigned_containers = {c["id"] for s in gen_best_solution.values() for c in s["container"]}
        if gen_best_fitness < best_fitness:
            best_solution = gen_best_solution
            best_fitness = gen_best_fitness
            best_generation = gen + 1
            best_population_index = gen_best_idx
        
        print(f"\nGeneration {gen+1} Stats:")
        print(f"- Best fitness: {format_rupiah(min(fitnesses))}")
        print(f"- Average fitness: {format_rupiah(sum(fitnesses)/len(fitnesses))}")
        print(f"- Current best fitness overall: {format_rupiah(best_fitness)} (Gen {best_generation}, Index {best_population_index})")
        
        # Lists to store details of genetic operations
        selection_details_list = []
        new_population = []
        crossover_details_list = []
        mutation_details_list = []
        
        # Selection & crossover loop
        selection_counter = 0
        while len(new_population) < POP_SIZE:
            # Select parents
            parent1, selection1 = rank_selection(population, fitnesses, selection_counter)
            selection_counter += 1
            parent2, selection2 = rank_selection(population, fitnesses, selection_counter)
            selection_counter += 1
            
            # Store selection details
            selection_details_list.append(selection1)
            selection_details_list.append(selection2)
            
            # Generate offspring with crossover
            offspring, crossover_details = crossover(
                parent1, parent2, 
                selection1["selected_idx"], selection2["selected_idx"]
            )
            crossover_details_list.append(crossover_details)
            
            # Apply mutation and store details
            mutated_offspring, mutation_details = mutate(offspring)
            mutation_details_list.append(mutation_details)
            
            # Add to new population
            new_population.append(mutated_offspring)
        
        # Display selection details
        display_selections(selection_details_list, gen+1)
        
        # Display crossover details
        display_crossovers(crossover_details_list, gen+1)
        
        # Display mutation details
        display_mutations(mutation_details_list, gen+1)
        
        # Calculate fitness of new population
        new_fitnesses = [evaluate_fitness(ind) for ind in new_population]
        display_chromosome_table(new_population, new_fitnesses, f"Generation {gen+1}: New Population After Genetic Operations")
        
        # Select best chromosomes for next generation
        population = sorted(new_population, key=evaluate_fitness)[:POP_SIZE]
    
    # Ensure we have a solution
    if best_solution is None:
        print("\n⚠️ No valid solution was found! Picking the best available solution.")
        best_solution = population[0] 
        best_fitness = evaluate_fitness(best_solution)

    # Display final best solution
    print("\n" + "="*80)
    print("FINAL BEST SOLUTION")
    print("="*80)
    print(f"✅ Selected from Generation {best_generation}, Expedition Index {best_population_index}")
    display_solution(best_solution, best_fitness)
    
    # Display best solution as chromosome
    best_chromosome = solution_to_chromosome(best_solution)
    print(f"\nBest Solution Chromosome: {' '.join(map(str, best_chromosome))}")
    print(f"Final Fitness: {format_rupiah(best_fitness)}")

if __name__ == "__main__":
    genetic_algorithm()