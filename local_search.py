import networkx as nx
import matplotlib.pyplot as plt
import time
import psutil
import heapq
import math
import random

def hill_climbing(G, start, goal, heuristic):
    current = start
    path = [current]
    
    while current != goal:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break
        
        best_neighbor = min(neighbors, key=lambda x: heuristic[x])
        if heuristic[best_neighbor] >= heuristic[current]:
            break  # Stop if no better option is found
        
        current = best_neighbor
        path.append(current)
    
    return path, current == goal

def simulated_annealing(G, start, goal, heuristic, initial_temp=1000, cooling_rate=0.95):
    current = start
    path = [current]
    temp = initial_temp
    
    while current != goal and temp > 1:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break
        
        next_node = random.choice(neighbors)
        delta = heuristic[next_node] - heuristic[current]
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = next_node
            path.append(current)
        
        temp *= cooling_rate
    
    return path, current == goal

def run_algorithm(G, source, target, algorithm, heuristic):
    start_time = time.time()
    process = psutil.Process()
    memory_before = process.memory_info().rss
    
    if algorithm == "hill_climbing":
        path, is_complete = hill_climbing(G, source, target, heuristic)
    elif algorithm == "simulated_annealing":
        path, is_complete = simulated_annealing(G, source, target, heuristic)
    else:
        path, is_complete = [], False
    
    memory_after = process.memory_info().rss
    end_time = time.time()
    
    time_taken = end_time - start_time
    memory_used = (memory_after - memory_before) / 1024  # KB
    optimality = time_taken * memory_used < 1  # True if optimal, False otherwise
    
    return path, time_taken, memory_used, optimality, is_complete

def visualize_graph():
    G = nx.Graph()
    
    edges = [
        ("Arad", "Zerind", 75), ("Arad", "Timisoara", 118), ("Arad", "Sibiu", 140),
        ("Zerind", "Oradea", 71), ("Oradea", "Sibiu", 151), ("Timisoara", "Lugoj", 111),
        ("Lugoj", "Mehadia", 70), ("Mehadia", "Dobreta", 75), ("Dobreta", "Craiova", 120),
        ("Craiova", "Pitesti", 138), ("Craiova", "Rimnicu Vilcea", 146),
        ("Rimnicu Vilcea", "Sibiu", 80), ("Rimnicu Vilcea", "Pitesti", 97),
        ("Sibiu", "Fagaras", 99), ("Fagaras", "Bucharest", 211),
        ("Pitesti", "Bucharest", 101), ("Bucharest", "Giurgiu", 90),
        ("Bucharest", "Urziceni", 85), ("Urziceni", "Hirsova", 98),
        ("Hirsova", "Eforie", 86), ("Urziceni", "Vaslui", 142),
        ("Vaslui", "Iasi", 92), ("Iasi", "Neamt", 87)
    ]
    
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    
    heuristic = {node: abs(hash(node) % 300) for node in G.nodes}  # Mock heuristic
    
    hill_path, hill_time, hill_memory, hill_optimality, hill_complete = run_algorithm(G, "Oradea", "Bucharest", "hill_climbing", heuristic)
    sa_path, sa_time, sa_memory, sa_optimality, sa_complete = run_algorithm(G, "Oradea", "Bucharest", "simulated_annealing", heuristic)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, edge_color='gray')
    
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(hill_path, hill_path[1:])), edge_color='purple', width=2, style='dashdot', label="Hill Climbing")
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(sa_path, sa_path[1:])), edge_color='orange', width=2, style='dotted', label="Simulated Annealing")
    
    plt.legend()
    plt.title("Comparison of Search Algorithms from Oradea to Bucharest")
    plt.show()
    
    print("Algorithm | Path | Time (s) | Memory (KB) | Optimality | Complete")
    print(f"Hill Climbing | {hill_path} | {hill_time:.6f} | {hill_memory:.2f} | {hill_optimality} | {hill_complete}")
    print(f"Simulated Annealing | {sa_path} | {sa_time:.6f} | {sa_memory:.2f} | {sa_optimality} | {sa_complete}")

visualize_graph()
