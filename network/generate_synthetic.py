import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

os.makedirs("power_law_53", exist_ok = True)
os.makedirs("skewed_53", exist_ok = True)

edges = []
idx = 0
for i in range(150):
    try:
        s = nx.utils.powerlaw_sequence(53, 1.2) #100 nodes, power-law exponent 2.5
        G = nx.expected_degree_graph(s, selfloops=False)
        a_power = nx.adjacency_matrix(G)
        degree_list = np.array(np.sum(a_power, axis = 0))[0]
        print(len(G.edges()), len(np.where(degree_list == 0)[0]) <= 1, min(degree_list))
        a_power = nx.adjacency_matrix(G)
        degree_list = np.array(np.sum(a_power, axis = 0))[0]
        hub_index_12 = np.argpartition(degree_list, -12)[-12:]
        np.save("power_law_53/a_{}.npy".format(idx), a_power)
        scipy.sparse.save_npz('power_law_53/a_{}.npz'.format(idx), a_power)
        np.save("power_law_53/perturb_node_hub_12_{}.npy".format(idx), hub_index_12)
        edges.append(len(G.edges()))
        idx += 1
    except:
        print("--")
        pass

# generate skewed
n = 53
edges = []
idx = 0
for _ in range(150):
    s = nx.utils.discrete_sequence(53, np.random.rand(53)) #100 nodes, power-law exponent 2.5
    s = np.array(s) / np.random.randint(3,8)
    G = nx.expected_degree_graph(s, selfloops=False)
    #print(len(G.edges()))
    a_power = nx.adjacency_matrix(G)
    degree_list = np.array(np.sum(a_power, axis = 0))[0]
    hub_index_12 = np.argpartition(degree_list, -12)[-12:]
    #np.save("power_law_53/a_{}.npy".format(idx), a_power)
    a_power = nx.adjacency_matrix(G)
    degree_list = np.array(np.sum(a_power, axis = 0))[0]
    print(len(G.edges()), len(np.where(degree_list == 0)[0]) <= 1, min(degree_list))
    scipy.sparse.save_npz('skewed_53/a_{}.npz'.format(idx), a_power)
    np.save("skewed_53/perturb_node_hub_12_{}.npy".format(idx), hub_index_12)
    edges.append(len(G.edges())) 
    idx += 1