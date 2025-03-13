import numpy as np
import random

# Parameter PSO
num_particles = 30
num_iterations = 100
w = 0.7  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient

# Spesifikasi mobil box
num_trucks = 4  # Jumlah mobil box
truck_capacity = 2000  # Kapasitas maksimum dalam kg
fuel_ratio = 0.2  # Konsumsi bahan bakar per km

# Barang yang akan dikirim
num_items = 10  # Jumlah barang
item_weights = np.random.randint(100, 500, size=num_items)  # Berat barang dalam kg
item_distances = np.random.randint(50, 500, size=num_items)  # Jarak tujuan dalam km
item_sizes = np.random.choice(["kecil", "menengah", "besar"], size=num_items)  # Ukuran barang

tariff_per_km = 500  # Tarif per km per kg

# Inisialisasi Partikel
particles = [np.random.randint(0, num_trucks, size=num_items) for _ in range(num_particles)]
velocities = [np.zeros(num_items) for _ in range(num_particles)]
pbest = particles.copy()
pbest_scores = [float('-inf')] * num_particles
gbest = None
gbest_score = float('-inf')

# Fungsi Evaluasi Fitness
def evaluate_fitness(particle):
    total_profit = 0
    truck_loads = [0] * num_trucks
    
    for i, truck in enumerate(particle):
        if truck_loads[truck] + item_weights[i] > truck_capacity:
            return float('-inf')  # Penalti jika kapasitas terlampaui
        truck_loads[truck] += item_weights[i]
        total_profit += item_weights[i] * item_distances[i] * tariff_per_km
    
    return total_profit

# Algoritma PSO
for iteration in range(num_iterations):
    for i in range(num_particles):
        fitness = evaluate_fitness(particles[i])
        
        if fitness > pbest_scores[i]:
            pbest_scores[i] = fitness
            pbest[i] = particles[i].copy()
        
        if fitness > gbest_score:
            gbest_score = fitness
            gbest = particles[i].copy()
    
    # Update velocity dan posisi
    for i in range(num_particles):
        r1, r2 = random.random(), random.random()
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest[i] - particles[i]) +
                         c2 * r2 * (gbest - particles[i]))
        
        particles[i] = np.round(particles[i] + velocities[i]).astype(int)
        particles[i] = np.clip(particles[i], 0, num_trucks - 1)
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Best Fitness = {gbest_score}")

# Cetak hasil terbaik
print("Optimal Truck Assignment:")
print(gbest)
