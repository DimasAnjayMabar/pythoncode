# ===============================================================
# Given dummy data type here

import math
from collections import Counter

# Training data format = [area / 1000, bedrooms, balcony, age], price
data = [
    ([1.2, 2, 0, 2], 500000), 
    ([2.3, 3, 2, 5], 620000),
    ([2.5, 4, 2, 1], 122500),
    ([3.65, 5, 3, 3], 6000000),
    ([1.8, 3, 1, 5], 2122000),
    ([3, 3, 1, 4], 120000), 
    ([1.222, 1, 0, 2], 450000), 
    ([4.6, 5, 3, 1], 6500000), 
    ([2.05, 2, 2, 2], 1530000),
    ([1.45, 2, 2, 3], 1563330)
]

# Questioned data format = [area / 1000, bedrooms, balcony, age], k
questioned_data = [
    ([2.4, 3, 0, 6], 2), 
    ([4.7, 5, 2, 1], 4),
    ([2.3, 2, 1, 3], 3),
    ([3.5, 3, 1, 5], 5),
    ([4.7, 3, 2, 3], 3),
    ([1.2, 1, 0, 7], 4),
    ([2.5, 3, 2, 4], 3),
    ([2.8, 2, 1, 9], 5)
]
# ===============================================================

# ===============================================================
# Functions
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def get_neighbors(data, patient, k):
    distances = [(train_point, label, euclidean_distance(patient, train_point)) for train_point, label in data]
    distances.sort(key=lambda x: x[2])
    return distances[:k]

def simple_knn_regression(neighbors):
    values = [label for _, label, _ in neighbors]
    return sum(values) / len(values)

def weighted_knn_regression(neighbors):
    weighted_sum = 0
    total_weight = 0
    for _, label, distance in neighbors:
        weight = 1 / distance if distance != 0 else float('inf')
        weighted_sum += label * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight != 0 else 0
# ===============================================================

# ===============================================================
# Run the main algorithm from the asked question here

for features, k in questioned_data:
    neighbors = get_neighbors(data, features, k)
    simple_result = simple_knn_regression(neighbors)
    weighted_result = weighted_knn_regression(neighbors)
    
    print(f"\ninput : {features}, K = {k}")
    print("Simple knn regression result : ", simple_result)
    print("Weighted knn regression result : ", weighted_result) 

# ===============================================================



