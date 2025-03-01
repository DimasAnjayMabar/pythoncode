import networkx as nx
import matplotlib.pyplot as plt
import time
import psutil
import heapq

def greedy_best_first_search(G, start, goal, heuristic):
    frontier = [(heuristic[start], start)]  # Priority queue based on heuristic
    came_from = {start: None}
    
    while frontier:
        _, current = heapq.heappop(frontier)
        
        if current == goal:
            break
        
        for neighbor in G.neighbors(current):
            if neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic[neighbor], neighbor))
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path

def run_algorithm(G, source, target, algorithm, heuristic=None):
    start_time = time.time()
    process = psutil.Process()
    memory_before = process.memory_info().rss
    
    if algorithm == "astar":
        path = nx.astar_path(G, source, target, weight="weight")
    elif algorithm == "dijkstra":
        path = nx.dijkstra_path(G, source, target, weight="weight")
    elif algorithm == "greedy":
        if heuristic is None:
            raise ValueError("Greedy search requires a heuristic function")
        path = greedy_best_first_search(G, source, target, heuristic)
    else:
        path = []
    
    memory_after = process.memory_info().rss
    end_time = time.time()
    
    return path, end_time - start_time, (memory_after - memory_before) / 1024  # Time in seconds, Memory in KB

def estimate_optimality(time_used, memory_used, dijkstra_time, dijkstra_memory):
    """
    Estimate how optimal an algorithm is compared to Dijkstra's optimal performance.

    Returns:
        str: "Optimal", "Near Optimal", or "Suboptimal"
    """
    time_ratio = time_used / dijkstra_time if dijkstra_time > 0 else float('inf')
    memory_ratio = memory_used / dijkstra_memory if dijkstra_memory > 0 else float('inf')

    if time_ratio <= 1.2 and memory_ratio <= 1.2:  # Within 20% of Dijkstra
        return "Optimal"
    elif time_ratio <= 2.0 and memory_ratio <= 2.0:  # Within 2x Dijkstra's cost
        return "Near Optimal"
    else:
        return "Suboptimal"

def visualize_graph():
    G = nx.Graph()
    
    # Define nodes and edges with distances
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
    
    # Simple heuristic: straight-line distance (mocked as direct edge weights for simplicity)
    heuristic = {node: abs(hash(node)) % 300 for node in G.nodes()}  

    # Compute paths
    astar_path, astar_time, astar_memory = run_algorithm(G, "Oradea", "Bucharest", "astar")
    dijkstra_path, dijkstra_time, dijkstra_memory = run_algorithm(G, "Oradea", "Bucharest", "dijkstra")
    greedy_path, greedy_time, greedy_memory = run_algorithm(G, "Oradea", "Bucharest", "greedy", heuristic)

    # Evaluate optimality
    astar_optimality = estimate_optimality(astar_time, astar_memory, dijkstra_time, dijkstra_memory)
    greedy_optimality = estimate_optimality(greedy_time, greedy_memory, dijkstra_time, dijkstra_memory)

    # Draw the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, edge_color='gray')
    
    # Highlight paths
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(astar_path, astar_path[1:])), edge_color='red', width=2, label="A*")
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(dijkstra_path, dijkstra_path[1:])), edge_color='blue', width=2, style='dashed', label="Dijkstra")
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(greedy_path, greedy_path[1:])), edge_color='green', width=2, style='dotted', label="Greedy")
    
    # Show weights
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.legend()
    plt.title("Comparison of A*, Dijkstra, and Greedy from Oradea to Bucharest")
    plt.show()
    
    # Print results
    print("\nAlgorithm | Path | Time (s) | Memory (KB)")
    print(f"A*        | {astar_path} | {astar_time:.6f} | {astar_memory:.2f}")
    print(f"Dijkstra  | {dijkstra_path} | {dijkstra_time:.6f} | {dijkstra_memory:.2f}")
    print(f"Greedy    | {greedy_path} | {greedy_time:.6f} | {greedy_memory:.2f}")

    print("\nPerformance-Based Optimality Check:")
    print(f"A*       -> {astar_optimality}")
    print(f"Dijkstra -> Optimal")  # Always optimal
    print(f"Greedy   -> {greedy_optimality}")

# Run the visualization
visualize_graph()
