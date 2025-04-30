# ===============================================================
# Given dummy data type here

import math
from collections import Counter

# Training data format = [sepal length, sepah width], species
data = [
    ([5.3, 3.7], 'Setosa'), 
    ([5.2, 3.8], 'Setosa'), 
    ([7.2, 3.0], 'Virginica'), 
    ([5.4, 3.4], 'Setosa'), 
    ([5.1, 3.3], 'Setosa'), 
    ([5.4, 3.9], 'Setosa'), 
    ([7.4, 2.8], 'Virginica'), 
    ([6.1, 2.8], 'Versicolor'), 
    ([7.3, 2.9], 'Virginica'), 
    ([6.0, 2.7], 'Versicolor'), 
    ([5.8, 2.8], 'Virginica'), 
    ([6.3, 2.3], 'Versicolor'), 
    ([5.1, 2.5], 'Versicolor'),
    ([6.3, 2.5], 'Versicolor'), 
    ([5.5, 2.4], 'Versicolor')    
]

# question format = [sepal lenght, sepal width], k
weighted_vote_question = [
    ([5.5, 2.3], 2), 
    ([5.7, 2.9], 4), 
    ([6.2, 2.7], 3), 
    ([6.6, 3.5], 5), 
    ([7.0, 3.9], 3), 
    ([5.1, 2.9], 4)
]

simple_vote_question = [
    ([5.2, 2.3], 2), 
    ([5.7, 2.5], 4), 
    ([6.9, 3.7], 3), 
    ([5.6, 2.4], 5), 
    ([7.3, 3.9], 3),
    ([5.1, 3.9], 4)
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

print("\n=== Weighted Majority Vote Questions ===")
for features, k in weighted_vote_question:
    neighbors = get_neighbors(data, features, k)
    result = weighted_majority_vote(neighbors)
    print(f"Input: {features}, K = {k} → Predicted Species: {result}")

print("\n=== Simple Majority Vote Questions ===")
for features, k in simple_vote_question:
    neighbors = get_neighbors(data, features, k)
    result = simple_majority_vote(neighbors)
    print(f"Input: {features}, K = {k} → Predicted Species: {result}")


# ===============================================================



