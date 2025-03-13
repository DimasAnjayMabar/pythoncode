import numpy as np
import random

# Parameter PSO
num_particles = 50
num_iterations = 200
w = 0.7  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient

# Spesifikasi perawat dan shift
num_nurses = 30  # Jumlah perawat
num_days = 30  # Jumlah hari dalam 1 bulan
shifts_per_day = 3  # Shift pagi, sore, malam

days_off_per_week = 2  # Minimal libur per minggu

# Ketersediaan perawat
nurse_experience = np.random.choice(["junior", "senior"], size=num_nurses)
nurse_certifications = np.random.choice(["ICU", "bayi", "umum"], size=num_nurses)

# Inisialisasi Partikel
particles = [np.random.randint(0, shifts_per_day + 1, size=(num_nurses, num_days)) for _ in range(num_particles)]
velocities = [np.zeros((num_nurses, num_days)) for _ in range(num_particles)]
pbest = particles.copy()
pbest_scores = [float('-inf')] * num_particles
gbest = None
gbest_score = float('-inf')

# Fungsi Evaluasi Fitness
def evaluate_fitness(particle):
    penalty = 0
    
    for nurse in range(num_nurses):
        days_off = np.sum(particle[nurse] == 0)
        
        if days_off < days_off_per_week * 4:
            penalty += 100  # Penalti jika libur kurang dari yang seharusnya
        
        for day in range(num_days - 1):
            if particle[nurse][day] > 0 and particle[nurse][day] == particle[nurse][day + 1]:
                penalty += 200  # Penalti jika perawat bekerja dua shift berturut-turut
    
    return -penalty  # Minimalkan penalti

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
        particles[i] = np.clip(particles[i], 0, shifts_per_day)
    
    if iteration % 20 == 0:
        print(f"Iteration {iteration}: Best Fitness = {gbest_score}")

# Cetak hasil terbaik
print("Optimal Nurse Schedule:")
print(gbest)
