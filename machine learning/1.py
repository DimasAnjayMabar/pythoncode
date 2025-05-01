# ===============================================================
# Given dummy data type here

import math
from collections import Counter

# Training data format = [fever, head aches, general aches, weakness, exhaustion, stuffy nose, sneezing, sore throat, chest discomfort], diagnosis
# Convert condition into = none = 0, mild = 1, severe = 2
data = [
    ([0, 1, 0, 0, 0, 1, 2, 2, 1], 'Cold'),
    ([2, 2, 2, 2, 2, 0, 0, 2, 2], 'Flu'),
    ([2, 2, 1, 2, 2, 2, 0, 2, 2], 'Flu'),
    ([0, 0, 0, 1, 0, 2, 0, 1, 1], 'Cold'),
    ([2, 2, 1, 2, 2, 2, 0, 2, 2], 'Flu'),
    ([0, 0, 0, 1, 0, 2, 2, 2, 0], 'Cold'),
    ([2, 2, 2, 2, 2, 0, 0, 0, 2], 'Flu'), 
    ([0, 0, 0, 0, 0, 2, 2, 0, 1], 'Cold')
]

# Patient in question 
patient_524 = [2, 1, 2, 1, 2, 0, 2, 0, 1]
# ===============================================================

# ===============================================================
# Functions

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def get_neighbors(data, patient, k):
    distances = [(train_point, label, euclidean_distance(patient, train_point)) for train_point, label in data]
    distances.sort(key=lambda x: x[2])
    return distances[:k]

def simple_majority_vote(neighbors):
    votes = [label for _, label, _ in neighbors]
    return Counter(votes).most_common(1)[0][0]

def weighted_majority_vote(neighbors):
    weights = {}
    for _, label, distance in neighbors:
        weight = 1 / distance if distance != 0 else float('inf')
        weights[label] = weights.get(label, 0) + weight
    return max(weights.items(), key=lambda x: x[1])[0]
# ===============================================================

# ===============================================================
# Run the main algorithm from the asked question here

for k in [3, 6, 4, 2, 5]:
    neighbors = get_neighbors(data, patient_524, k)
    simple_result = simple_majority_vote(neighbors)
    weighted_result = weighted_majority_vote(neighbors)
    
    print(f"\nK = {k}")
    print("Simple majority vote result : ", simple_result)
    print("Weighted majority vote result : ", weighted_result) 

# ===============================================================



