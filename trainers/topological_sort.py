# Code adapted from https://www.geeksforgeeks.org/python-program-for-topological-sorting/


# Python program to print topological sorting of a DAG 
from collections import defaultdict 
  
# Class to represent a graph 
class Graph: 
    def __init__(self, vertices): 
        self.graph = defaultdict(list) # dictionary containing adjacency List 
        self.V = vertices # No. of vertices 
  
    # function to add an edge to graph 
    def addEdge(self, u, v): 
        self.graph[u].append(v) 
  
    # A recursive function used by topologicalSort 
    def topologicalSortUtil(self, v, visited, stack): 
  
        # Mark the current node as visited. 
        visited[v] = True
  
        # Recur for all the vertices adjacent to this vertex 
        for i in self.graph[v]: 
            if visited[i] == False: 
                self.topologicalSortUtil(i, visited, stack) 
  
        # Push current vertex to stack which stores result 
        stack.insert(0, v) 
  
    # The function to do Topological Sort. It uses recursive  
    # topologicalSortUtil() 
    def topologicalSort(self, assert_head=None):

        if assert_head is not None:
            for v in list(self.graph.keys()):
                if v != assert_head:
                    if v not in self.graph[assert_head]:
                        self.graph[assert_head].insert(0, v)
            pass

        # Mark all the vertices as not visited 
        visited = [False] * self.V 
        stack = [] 
  
        # Call the recursive helper function to store Topological 
        # Sort starting from all vertices one by one 
        for i in range(self.V):
            if assert_head is None:
                if visited[i] == False: 
                    self.topologicalSortUtil(i, visited, stack) 
            else:
                if visited[i] == False and i != assert_head:
                    self.topologicalSortUtil(i, visited, stack)

        if assert_head is not None:
            if assert_head in stack:
                head_idx = stack.index(assert_head)
                del stack[head_idx]
            self.topologicalSortUtil(assert_head, visited, stack)
  
        # Print contents of stack 
        # print(stack)

        if assert_head is not None:
            assert stack[0] == assert_head, "Asserting head failed"
            # if stack[0] != assert_head:
            #     print(assert_head, stack)

        return stack


if __name__ == "__main__":
    g = Graph(5)
    g.addEdge(4, 2)
    g.addEdge(4, 0)
    g.addEdge(3, 0)
    g.addEdge(3, 1)
    g.addEdge(2, 3)
    g.addEdge(3, 1)
      
    print("Following is a Topological Sort of the given graph")
    res = g.topologicalSort(assert_head=2)
    print(res)
