import numpy as np
import random

# Parameter
num_units = 7  # Jumlah pembangkit listrik
num_periods = 12  # Jumlah periode dalam setahun
num_ants = 10  # Jumlah semut
num_iterations = 4  # Jumlah iterasi yang diminta
evaporation_rate = 0.4  # Evaporasi feromon
reserve_capacity = 15  # Kapasitas cadangan MW

# Kapasitas masing-masing pembangkit
unit_capacity = [20, 15, 35, 40, 15, 15, 10]
# Interval maintenance dalam 1/2 tahun (1 tahun = 12 bulan)
maintenance_interval = [2, 2, 1, 1, 1, 2, 1]

# Inisialisasi feromon
tau = np.ones((num_units, num_periods))

# Fungsi untuk membangun jadwal maintenance
def construct_schedule(pheromone):
    schedule = np.zeros((num_units, num_periods))
    
    for unit in range(num_units):
        possible_periods = list(range(num_periods))
        random.shuffle(possible_periods)  # Randomisasi pilihan
        
        for _ in range(maintenance_interval[unit]):
            chosen_period = max(possible_periods, key=lambda p: pheromone[unit][p])
            schedule[unit][chosen_period] = 1  # 1 berarti unit ini dalam maintenance
            possible_periods.remove(chosen_period)
    
    return schedule

# Fungsi evaluasi solusi
def evaluate_schedule(schedule):
    for period in range(num_periods):
        active_capacity = sum(unit_capacity[i] for i in range(num_units) if schedule[i][period] == 0)
        if active_capacity < 100 - reserve_capacity:  # Tidak boleh kurang dari beban minimum
            return float('inf')
    return sum(sum(schedule))  # Semakin kecil maintenance total semakin baik

# Fungsi update feromon
def update_pheromone(pheromone, best_schedule):
    pheromone *= (1 - evaporation_rate)  # Evaporasi
    for unit in range(num_units):
        for period in range(num_periods):
            if best_schedule[unit][period] == 1:
                pheromone[unit][period] += 1  # Tambah feromon untuk jadwal terbaik

# Algoritma ACO
best_schedule = None
best_cost = float('inf')

for iteration in range(num_iterations):
    print(f"Iteration {iteration+1}")
    for _ in range(num_ants):
        schedule = construct_schedule(tau)
        cost = evaluate_schedule(schedule)
        if cost < best_cost:
            best_cost = cost
            best_schedule = schedule.copy()
    
    update_pheromone(tau, best_schedule)
    print(f"Best cost so far: {best_cost}")

# Cetak hasil jadwal optimal
print("Optimal Maintenance Schedule:")
print(best_schedule)
