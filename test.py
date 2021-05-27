import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G=nx.DiGraph()

# a list of nodes:
pages = ["1","2","3","4"]
G.add_nodes_from(pages)

print("Nodes of graph: ")
print(G.nodes())
print("Edges of graph: ")
print(G.edges())

G.add_edges_from([('1','2'), ('1','4'),('1','3'), ('4','1'),('2','3'),('2','4'),('3','1'),('4','3')])
print("Nodes of graph: ")
print(G.nodes())
print("Edges of graph: ")
print(G.edges())

# The function G.out_edges('node') returns the numbers of nodes 'node' links to.

print("Number of outward links for each node:")
for page in pages:
    print(["Page %s = %s"% (page,str(len(G.out_edges(page))))])

nx.draw(G, with_labels = True)
plt.show() # display