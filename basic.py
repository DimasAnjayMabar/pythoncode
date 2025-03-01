import time
import tracemalloc
from queue import PriorityQueue
import networkx as nx
import matplotlib.pyplot as plt

def measure_performance(func, *args):
    tracemalloc.start()
    start_time = time.time()
    
    result = func(*args[:func.__code__.co_argcount])
    
    end_time = time.time()
    memory_used = tracemalloc.get_traced_memory()[1] 
    tracemalloc.stop()
    
    exec_time = end_time - start_time
    return result, exec_time, memory_used

def bfs(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        node, path = queue.pop(0)
        if node == goal:
            return path
        for neighbor in graph[node]:
            if isinstance(neighbor, tuple):  
                neighbor = neighbor[0] 
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return None

def dfs(graph, start, goal, path=None):
    if path is None:
        path = [start]
    if start == goal:
        return path
    for neighbor in graph[start]:
        if isinstance(neighbor, tuple):
            neighbor = neighbor[0]
        if neighbor not in path:
            new_path = dfs(graph, neighbor, goal, path + [neighbor])
            if new_path:
                return new_path
    return None

def uniform_cost(graph, start, goal):
    pq = PriorityQueue()
    pq.put((0, start, [start]))
    visited = set()

    while not pq.empty():
        cost, node, path = pq.get()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph.get(node, []):
            if neighbor not in path:
                pq.put((cost + weight, neighbor, path + [neighbor]))
    return None

def depth_limited(graph, start, goal, limit, path=None):
    if path is None:
        path = [start]
    if start == goal:
        return path
    if limit <= 0:
        return None
    for neighbor in graph[start]:
        if isinstance(neighbor, tuple):
            neighbor = neighbor[0]
        if neighbor not in path:
            new_path = depth_limited(graph, neighbor, goal, limit - 1, path + [neighbor])
            if new_path:
                return new_path
    return None

def iterative_deepening(graph, start, goal, max_depth):
    for depth in range(max_depth):
        path = depth_limited(graph, start, goal, depth)
        if path:
            return path
    return None

def visualize_graph(graph, path):
    G = nx.DiGraph()

    for node in graph:
        for neighbor in graph[node]:
            if isinstance(neighbor, tuple): 
                G.add_edge(node, neighbor[0], weight=neighbor[1])
            else: 
                G.add_edge(node, neighbor)

    pos = nx.spring_layout(G)  
    plt.figure(figsize=(8, 6))

    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color='gray', node_size=2000, font_size=12)

    labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True) if 'weight' in d}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

    if path:
        edges = [(path[i], path[i+1]) for i in range(len(path)-1) if path[i+1] in graph.get(path[i], [])]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red', node_size=2000)

    plt.title("Visualisasi Graph dan Path")
    plt.show()

simple_graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': ['H'],
    'F': [],
    'G': ['I'],
    'H': [],
    'I': []
}

weighted_graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 3), ('G', 6)],
    'D': [],
    'E': [('H', 1)],
    'F': [],
    'G': [('I', 2)],
    'H': [],
    'I': []
}

start, goal = 'A', 'I'

algorithms = [bfs, dfs, depth_limited, iterative_deepening]
for algorithm in algorithms:
    if algorithm == depth_limited or algorithm == iterative_deepening:
        path, exec_time, memory_used = measure_performance(algorithm, simple_graph, start, goal, 5)
    else:
        path, exec_time, memory_used = measure_performance(algorithm, simple_graph, start, goal)
    
    print(f"{algorithm.__name__}: Path: {path}, Time: {exec_time:.6f}s, Memory: {memory_used / 1024:.2f} KB")
    visualize_graph(simple_graph, path)

path, exec_time, memory_used = measure_performance(uniform_cost, weighted_graph, start, goal)
print(f"uniform_cost: Path: {path}, Time: {exec_time:.6f}s, Memory: {memory_used / 1024:.2f} KB")
visualize_graph(weighted_graph, path)
