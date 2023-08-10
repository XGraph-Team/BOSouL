import networkx as nx
import torch_geometric.datasets as ds
import ndlib.models.ModelConfig as mc

from torch_geometric.datasets import Planetoid

def connSW(n):
    g = nx.connected_watts_strogatz_graph(n, 20, 0.1)
    while nx.is_connected(g) == False:
        g = nx.connected_watts_strogatz_graph(n, 20, 0.1)

    return g

def BA(n):
    g = nx.barabasi_albert_graph(n, 10)

    return g

def ER(n):

    g = nx.erdos_renyi_graph(n, 0.02)

    while nx.is_connected(g) == False:
        g = nx.erdos_renyi_graph(n, 0.02)

    return g



def CiteSeer():
    dataset = Planetoid(root='./Planetoid', name='CiteSeer')  # Cora, CiteSeer, PubMed
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)

    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute=None)

    return g

def PubMed():
    dataset = Planetoid(root='./Planetoid', name='PubMed')  # Cora, CiteSeer, PubMed
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)

    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute=None)

    return g

def Cora():
    dataset = Planetoid(root='./Planetoid', name='Cora')  # Cora, CiteSeer, PubMed
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)

    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute=None)

    return g

def photo():

    dataset = ds.Amazon(root='./geo', name = 'Photo')
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)
    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute=None)

    return g

def coms():

    dataset = ds.Amazon(root='./geo', name = 'Computers')
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)
    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute=None)

    return g