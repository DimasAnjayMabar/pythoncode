import numpy as np
import random

# Parameter ACO
num_ants = 20
num_iterations = 4
evaporation_rate = 0.4
alpha = 1  # Pengaruh feromon
beta = 2  # Pengaruh jarak

# Kota-kota dan jaraknya (diambil dari data jarak antar kota di Jawa Timur)
cities = ["Surabaya", "Lamongan", "Sidoarjo", "Malang", "Jombang", "Mojokerto", "Probolinggo", "Situbondo", "Jember", "Banyuwangi", "Lumajang", "Blitar", "Kediri"]
num_cities = len(cities)
distance_matrix = np.random.randint(50, 500, size=(num_cities, num_cities))
np.fill_diagonal(distance_matrix, 0)  # Jarak dari kota ke dirinya sendiri = 0

# Inisialisasi feromon
tau = np.ones((num_cities, num_cities))

# Fungsi untuk membangun rute
def construct_route(pheromone, distance_matrix):
    route = [random.randint(0, num_cities - 1)]  # Mulai dari kota acak
    
    while len(route) < num_cities:
        current_city = route[-1]
        probabilities = (pheromone[current_city] ** alpha) * ((1 / distance_matrix[current_city]) ** beta)
        probabilities[route] = 0  # Hindari kembali ke kota yang sudah dikunjungi
        next_city = np.random.choice(range(num_cities), p=probabilities / probabilities.sum())
        route.append(next_city)
    
    return route

# Fungsi evaluasi jarak total
def evaluate_route(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route) - 1))

# Fungsi update feromon
def update_pheromone(pheromone, best_route, distance_matrix):
    pheromone *= (1 - evaporation_rate)  # Evaporasi
    for i in range(len(best_route) - 1):
        pheromone[best_route[i], best_route[i+1]] += 1 / distance_matrix[best_route[i], best_route[i+1]]

# Algoritma ACO
best_route = None
best_distance = float('inf')

for iteration in range(num_iterations):
    print(f"Iteration {iteration+1}")
    for _ in range(num_ants):
        route = construct_route(tau, distance_matrix)
        distance = evaluate_route(route, distance_matrix)
        if distance < best_distance:
            best_distance = distance
            best_route = route.copy()
    
    update_pheromone(tau, best_route, distance_matrix)
    print(f"Best distance so far: {best_distance}")

# Cetak hasil rute optimal
print("Optimal Truck Route:")
print([cities[i] for i in best_route])
