import random

##Implement of the graph
class Node:
    def __init__(self, id):
        self.id = id
        self.ANode = []  # List of adjacent nodes

class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, edge):
        self.edges.append(edge)
        edge.node1.ANode.append(edge.node2)
        edge.node2.ANode.append(edge.node1)

#Implementation of the 3 functions 
class GraphMetrics:                           
    def __init__(self, graph):                                  # Initialize GraphMetrics with the given graph.  
        self.graph = graph

    def degree_centrality(self):                                   #Calculate the degree centrality (number of outgoing edges) of a vertex.
        degree_centrality = {}
        for vertex in self.graph.nodes:
            degree_centrality[vertex] = len(self.graph.nodes[vertex].ANode)
        return degree_centrality

    def closeness_centrality(self):                                  #  Calculate the closeness centrality of a vertex.
        closeness_centrality = {}
        for vertex in self.graph.nodes:
            total_distance = 0
            visited = set()
            queue = [(vertex, 0)]

            while queue:
                v, distance = queue.pop(0)
                if v not in visited:
                    visited.add(v)
                    total_distance += distance
                    for neighbor in self.graph.nodes[v].ANode:
                        queue.append((neighbor.id, distance + 1))

            closeness_centrality[vertex] = 1 / total_distance if total_distance != 0 else 0

        return closeness_centrality

    def betweenness_centrality(self, agent):                                             #Calculate the betweenness centrality of a vertex.
        betweenness_centrality = {}
        for vertex in self.graph.nodes:
            betweenness = 0
            for s in self.graph.nodes:
                for t in self.graph.nodes:
                    if s != t:
                        shortest_paths = agent.compute_all_shortest_paths()
                        count_st = self.count_shortest_paths(s, t, shortest_paths[s])
                        count_st_v = self.count_shortest_paths(s, t, agent.compute_all_shortest_paths()[s][t])
                        betweenness += count_st_v / count_st if count_st != 0 else 0

            betweenness_centrality[vertex] = betweenness

        return betweenness_centrality
    
    def bfs_shortest_paths(self, source):                                    #Using Breadth – First Search (BFS) to find shortest distance 
        visited = set()
        distance = {v: float('inf') for v in self.graph.nodes}
        paths = {v: [] for v in self.graph.nodes}
        distance[source] = 0
        queue = [(source, [])]

        while queue:
            v, path = queue.pop(0)
            if v not in visited:
                visited.add(v)
                paths[v].append(path + [v])
                for neighbor in self.graph.nodes[v].ANode:
                    if distance[neighbor.id] >= distance[v] + 1:
                        distance[neighbor.id] = distance[v] + 1
                        queue.append((neighbor.id, path + [v]))

        return paths

    def count_shortest_paths(self, s, t, paths):        #Count number of shortest path taken between two nodes where s and t are the nodes       
        count = 0
        if isinstance(paths, dict):
            if s in paths and t in paths[s]:
                count += 1
        elif isinstance(paths, list):  
            if t in paths:
                count += 1
        return count

#Agent Deisgn and Movement
class GraphAgent:                                    # class GraphAgent
    def __init__(self, graph):                       #Function defined innt to initialize graph , Graph metrics also defined for the agent to move and also store the memory of the movement
        self.graph = graph
        self.graph_metrics = GraphMetrics(graph)        # Create an instance of GraphMetrics
        self.memory = []                                # Memory to store the value of nodes

    def compute_all_shortest_paths(self):               #Calculate shortest paths between all the pair of nodes in the graph            
        shortest_paths = {}
        for start_node in self.graph.nodes:
            shortest_paths[start_node] = {}
            for target_node in self.graph.nodes:
                shortest_paths[start_node][target_node] = self.shortest_path_walk(start_node, target_node)
        return shortest_paths

    def current_state(self, start_node, target_node):              #tuple containing current state of the agent
        return (start_node, target_node)

    def agent_nodes(self, visited_nodes):                 # Stroing the visiting the nodes during the agent movement in  the graph
        self.memory.append(visited_nodes)

    def random_walk(self, start_node, target_node):                 #Agent performing random walk in the graph
        current_node = start_node
        visited_nodes = [start_node]

        while current_node != target_node:                     #if agent didnt reach the target node 
            Anode = self.graph.nodes[current_node].ANode
            if not Anode:
                print("Agent cannot move as there are no adjacent nodes.")            # if there is no adjacent node 
                break
            next_node = random.choice(Anode).id                         #next node is a random node when agent is moving in the graph 
            visited_nodes.append(next_node)
            current_node = next_node

        self.agent_nodes(visited_nodes)                        # Store visited nodes in memory
        return visited_nodes

    def shortest_path_walk(self, start_node, target_node):                 #Function for the shortest path taken by the agent to move from start node to target node
        visited = set()
        queue = [(start_node, [start_node])]

        while queue:
            current_node, path = queue.pop(0)

            if current_node == target_node:
                return path

            if current_node not in visited:
                visited.add(current_node)

                for neighbor in self.graph.nodes[current_node].ANode:
                    if neighbor.id not in visited:
                        queue.append((neighbor.id, path + [neighbor.id]))

        return []
    

#Simulation 
class World:
    def __init__(self, graph, agent):
        self.graph = graph
        self.agent = agent

    def run_simulations(self, num_episodes):
        random_walk_results = []
        shortest_path_results = []

        for _ in range(num_episodes):
            start_node = random.randint(1, len(self.graph.nodes))
            target_node = random.randint(1, len(self.graph.nodes))
            while start_node == target_node:
                target_node = random.randint(1, len(self.graph.nodes))

            visited_nodes_random = self.agent.random_walk(start_node, target_node)
            random_walk_results.append(len(set(visited_nodes_random)))

            shortest_path = self.agent.shortest_path_walk(start_node, target_node)
            shortest_path_results.append(len(set(shortest_path)))

        return random_walk_results, shortest_path_results

# Example usage
graph = Graph()
for i in range(1, 8):
    graph.add_node(Node(i))

edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1)]
for edge in edges:
    graph.add_edge(Edge(graph.nodes[edge[0]], graph.nodes[edge[1]]))

agent = GraphAgent(graph)
world = World(graph, agent)
random_walk_results, shortest_path_results = world.run_simulations(1000)          #1000 simulation data

print("Random Walk Results (First 10):", random_walk_results[:10])                     #Running only 10 at every program
print("Shortest Path Results (First 10):", shortest_path_results[:10])

avg_random_walk = sum(random_walk_results) / len(random_walk_results)                    #calculating the average of each and every random move
avg_shortest_path = sum(shortest_path_results) / len(shortest_path_results)              #calculating the average of each and every shortest path taken
print(f"Average number of visited nodes in Random Walk: {avg_random_walk:.3f}")
print(f"Average number of visited nodes in Shortest Path: {avg_shortest_path:.3f}")

#Printing the centralities
graph_metrics = GraphMetrics(graph)
degree_centrality = graph_metrics.degree_centrality()
closeness_centrality = graph_metrics.closeness_centrality()
betweenness_centrality = graph_metrics.betweenness_centrality(agent)

print("Degree Centrality:", degree_centrality)
print("Closeness Centrality:", closeness_centrality)
print("Betweenness Centrality:", betweenness_centrality)