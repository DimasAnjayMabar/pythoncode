# ===============================================================
# Given dummy data type here

import math
from collections import Counter

# Training data format = [age, gender (male = 0, female = 1), height], weight
data = [
    ([35, 1, 1.52], 50), 
    ([52, 0, 1.77], 115), 
    ([45, 0, 1.83], 96), 
    ([70, 1, 1.55], 41),
    ([24, 0, 1.82], 79), 
    ([43, 0, 1.89], 109),
    ([68, 0, 1.76], 73),
    ([77, 1, 1.71], 104),
    ([45, 1, 1.74], 64),
    ([28, 0, 1.78], 136)
]

# Questioned data format = ([age, gender, height], k)
questioned_data = [
    ([55, 0, 1.63], 2), 
    ([37, 0, 1.49], 4),
    ([23, 1, 1.75], 3),
    ([30, 1, 1.65], 5),
    ([32, 0, 1.53], 3),
    ([40, 0, 1.70], 4),
    ([55, 1, 1.80], 3),
    ([28, 1, 1.90], 5)
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



