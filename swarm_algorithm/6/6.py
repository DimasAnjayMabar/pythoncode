import numpy as np
import random

# Parameter PSO
num_particles = 50
num_iterations = 200
w = 0.7  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient

# Spesifikasi mesin produksi
num_machines = 4  # Jumlah mesin produksi
machine_available = np.ones(num_machines)  # 1 jika tersedia, 0 jika maintenance

# Spesifikasi barang
num_products = 12  # Jumlah jenis barang
product_prices = np.random.randint(500, 5000, size=num_products)  # Harga jual barang
production_costs = np.random.randint(100, 2000, size=num_products)  # Biaya produksi
production_times = np.random.randint(1, 10, size=num_products)  # Waktu produksi dalam jam

# Kapasitas produksi mesin
machine_capacity = np.random.randint(20, 50, size=num_machines)  # Kapasitas produksi tiap mesin

total_production_hours = 40  # Total jam kerja per minggu

# Inisialisasi Partikel
particles = [np.random.randint(0, num_products, size=num_machines) for _ in range(num_particles)]
velocities = [np.zeros(num_machines) for _ in range(num_particles)]
pbest = particles.copy()
pbest_scores = [float('-inf')] * num_particles
gbest = None
gbest_score = float('-inf')

# Fungsi Evaluasi Fitness
def evaluate_fitness(particle):
    total_profit = 0
    machine_usage = [0] * num_machines
    
    for i in range(num_machines):
        if machine_available[i] == 0:
            return float('-inf')  # Penalti jika mesin tidak tersedia
        
        product = particle[i]
        if machine_usage[i] + production_times[product] > total_production_hours:
            return float('-inf')  # Penalti jika melebihi kapasitas
        
        machine_usage[i] += production_times[product]
        total_profit += product_prices[product] - production_costs[product]
    
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
        particles[i] = np.clip(particles[i], 0, num_products - 1)
    
    if iteration % 20 == 0:
        print(f"Iteration {iteration}: Best Fitness = {gbest_score}")

# Cetak hasil terbaik
print("Optimal Production Schedule:")
print([gbest[i] for i in range(num_machines)])
