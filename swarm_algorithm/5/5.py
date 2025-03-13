import numpy as np
import random

# Parameter PSO
num_particles = 50
num_iterations = 200
w = 0.7  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient

# Spesifikasi bus
num_buses = 15  # Jumlah bus
bus_available = np.ones(num_buses)  # 1 jika tersedia, 0 jika maintenance

# Spesifikasi sopir dan kondektur
num_drivers = 20
num_conductors = 20
drivers_available = np.ones(num_drivers)  # 1 jika tersedia, 0 jika cuti
conductors_available = np.ones(num_conductors)

# Rute kota besar di Pulau Jawa
cities = ["Jakarta", "Bandung", "Semarang", "Yogyakarta", "Surabaya", "Malang", "Solo", "Cirebon", "Tegal", "Purwokerto", "Magelang", "Blitar", "Madiun", "Kediri", "Pasuruan"]
num_routes = len(cities)
distance_matrix = np.random.randint(50, 600, size=(num_routes, num_routes))
np.fill_diagonal(distance_matrix, 0)

# Inisialisasi Partikel
particles = [np.random.randint(0, num_routes, size=num_buses) for _ in range(num_particles)]
velocities = [np.zeros(num_buses) for _ in range(num_particles)]
pbest = particles.copy()
pbest_scores = [float('-inf')] * num_particles
gbest = None
gbest_score = float('-inf')

# Fungsi Evaluasi Fitness
def evaluate_fitness(particle):
    total_idle = 0
    total_empty_distance = 0
    
    for i in range(num_buses):
        if bus_available[i] == 0:
            return float('-inf')  # Penalti jika bus tidak tersedia
        
        if random.random() > 0.8:
            total_idle += 5  # Simulasi waktu idle
        
        if i > 0 and particle[i] == particle[i-1]:
            total_empty_distance += distance_matrix[particle[i], particle[i-1]]
    
    return -(total_idle + total_empty_distance)  # Minimalkan waktu idle dan jarak kosong

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
        particles[i] = np.clip(particles[i], 0, num_routes - 1)
    
    if iteration % 20 == 0:
        print(f"Iteration {iteration}: Best Fitness = {gbest_score}")

# Cetak hasil terbaik
print("Optimal Bus Schedule:")
print([cities[i] for i in gbest])
