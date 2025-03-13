import numpy as np
import random

# Parameter PSO
num_particles = 30
num_iterations = 100
w = 0.7  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient

# Kapal dan peti kemas
num_containers = 10  # Jumlah peti kemas
num_slots = 10  # Kapasitas tempat di kapal
container_weights = np.random.randint(5, 50, size=num_containers)  # Bobot peti kemas
container_types = np.random.choice(["biasa", "khusus"], size=num_containers)  # Jenis peti
shipping_cost_per_ton = 1000  # Biaya per tonase

# Inisialisasi Partikel
particles = [np.random.permutation(num_slots) for _ in range(num_particles)]
velocities = [np.zeros(num_slots) for _ in range(num_particles)]
pbest = particles.copy()
pbest_scores = [float('-inf')] * num_particles
gbest = None
gbest_score = float('-inf')

# Fungsi Evaluasi Fitness
def evaluate_fitness(particle):
    total_cost = 0
    for i, slot in enumerate(particle):
        if slot >= num_containers:
            continue
        total_cost += container_weights[slot] * shipping_cost_per_ton
        
        # Penalti jika aturan dilanggar (misal, peti berat di atas peti ringan)
        if i > 0 and container_weights[particle[i]] > container_weights[particle[i-1]]:
            total_cost -= 5000  # Penalti
    return total_cost

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
        particles[i] = np.clip(particles[i], 0, num_slots - 1)
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Best Fitness = {gbest_score}")

# Cetak hasil terbaik
print("Optimal Container Placement:")
print(gbest)
