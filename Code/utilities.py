import pandas as pd
import networkx as nx
import numpy as np
import itertools
from collections import defaultdict
import random
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

def read_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test

def build_graph(edges):
    g = nx.Graph()
    g.add_edges_from(edges.values)
    return g

def common_neighbor(graph, test_edges):
    predictions = []
    for idx, row in test_edges.iterrows():
        node1, node2 = row['node1'], row['node2']
        if graph.has_node(node1) and graph.has_node(node2):
            common_neighbors = len(list(nx.common_neighbors(graph, node1, node2)))
            predictions.append((node1, node2, common_neighbors))
        else:
            predictions.append((node1, node2, 0))

    return predictions

#0.005 4: 0.929
#0.005 2: 0.839
def katz_index(g, beta=0.005, max_path=4):
    nodes = g.nodes()
    adj_matrix = np.zeros((len(nodes), len(nodes)))
    node_index = {node: idx for idx, node in enumerate(nodes)}

    for l in range(1, max_path + 1):
        for path in nx.all_pairs_shortest_path(g, cutoff=l):
            node_u = node_index[path[0]]
            for node_v in path[1]:
                if len(path[1][node_v]) == l:
                    adj_matrix[node_u][node_index[node_v]] += beta ** l

    return adj_matrix, node_index

def katz_predict(adj_matrix, node_index, test):
    predictions = []
    for _, row in test.iterrows():
        node_u_idx = node_index[row['node1']]
        node_v_idx = node_index[row['node2']]
        score = adj_matrix[node_u_idx][node_v_idx]
        predictions.append((row['node1'], row['node2'], score))
    return predictions


#alpha = 0.5 scores = 0.857
#alpha = 0.85 scores = 0.851
#alpha = 0.3 scores = 0.861
#alpha = 0.2 scores = 0.861
def page_rank(g, personalization, alpha=0.2):
    pagerank_scores = nx.pagerank(g, alpha=alpha, personalization=personalization)

    predictions = []
    nodes = list(g.nodes())
    for node1, node2 in itertools.product(nodes, repeat=2):
        score1 = pagerank_scores[node1]
        score2 = pagerank_scores[node2]
        combined_score = score1 * score2
        predictions.append((node1, node2, combined_score))

    return predictions

#seed = 3 0.929
def random_walks(G, num_walks, walk_length, seed=3):
    random.seed(seed)
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                next_node = random.choice(list(G.neighbors(cur)))
                walk.append(next_node)
            walks.append(walk)
    return walks

# 64 6 0 1 3 92.9
# 64 7 1 0 3 94.1
# 64 6 0 0 3 93.5
# 64 8 2 0 3 94.1
def deepwalk_model(walks, vector_size=64, window=7, min_count=1, sg=0, seed=3):

    model = Word2Vec(walks, vector_size=vector_size, window=window, min_count=min_count, sg=sg, seed=seed)
    return model

def generate_predictions(model, test_edges):
    predictions = []
    for _, row in test_edges.iterrows():
        try:
            node1_vec = model.wv[row['node1']]
            node2_vec = model.wv[row['node2']]
            score = np.dot(node1_vec, node2_vec)
        except KeyError:
            score = 0
        predictions.append((row['node1'], row['node2'], score))
    return predictions

def visualize_top_edges(top_edges_filepath):
    top_edges = pd.read_csv(top_edges_filepath, header=None, names=['node1', 'node2'])
    G = nx.Graph()

    for index, row in top_edges.iterrows():
        G.add_edge(row['node1'], row['node2'])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', linewidths=1, font_size=4)

    plt.title("Top 100 Predicted Edges")
    plt.show()

def predicted_file(predictions, test_edges, output_filepath, top_edges_filepath):

    predictions_df = pd.DataFrame(predictions, columns=['node1', 'node2', 'score'])
    test_with_scores = test_edges.merge(predictions_df, on=['node1', 'node2'])
    top_100_idx = test_with_scores['score'].nlargest(100).index
    result = pd.DataFrame({
        'EdgeID': range(1, len(test_with_scores) + 1),
        'label': 0
    })
    result.loc[top_100_idx, 'label'] = 1
    result.to_csv(output_filepath, index=False)

    top_100_edges = test_with_scores.loc[top_100_idx, ['node1', 'node2']]
    top_100_edges.to_csv(top_edges_filepath, index=False, header=False)

def predicted_file_final(predictions, test_edges, top_edges_filepath):
    predictions_df = pd.DataFrame(predictions, columns=['node1', 'node2', 'score'])
    test_with_scores = test_edges.merge(predictions_df, on=['node1', 'node2'])
    top_100_idx = test_with_scores['score'].nlargest(100).index
    result = pd.DataFrame({
        'EdgeID': range(1, len(test_with_scores) + 1),
        'label': 0
    })

    top_100_edges = test_with_scores.loc[top_100_idx, ['node1', 'node2']]
    top_100_edges.to_csv(top_edges_filepath, index=False, header=False)