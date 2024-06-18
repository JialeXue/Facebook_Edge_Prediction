from utilities import *
import pandas as pd
import os

def main():
    train_path = 'trainingset.csv'
    test_path = 'testset.csv'
    train, test = read_data(train_path, test_path)
    ktrain = pd.read_csv(train_path, header=None, names=['node1', 'node2'])
    ktest = pd.read_csv(test_path, header=None, names=['node1', 'node2'])
    g_train = build_graph(train)

    cn_results = common_neighbor(g_train, ktest)
    # predicted_file(cn_results, ktest, 'Results/commonNeighbors.csv', 'Topscn.csv')

    os.makedirs('Results', exist_ok=True)
    os.makedirs('Tops', exist_ok=True)

    adj_matrix, node_index = katz_index(g_train)
    katz_results = katz_predict(adj_matrix, node_index, ktest)
    predicted_file(katz_results, ktest, 'Results/katzPara.csv', 'Tops/katz.csv')

    personalization = {node: 1.0 for node in g_train.nodes()}
    pr_results = page_rank(g_train, personalization)
    predicted_file(pr_results, ktest, 'Results/pageRankPara.csv', 'Tops/pageRank.csv')

    #deepwalk4 highest: 93.1
    dw_train = build_graph(ktrain)
    walks = random_walks(dw_train, num_walks=8, walk_length=80)
    model = deepwalk_model(walks)
    predictions = generate_predictions(model, ktest)
    predicted_file(predictions, ktest, 'Results/randomwalk.csv', 'Tops/deepwalk.csv')

def main_final():
    train_path = 'trainingset.csv'
    test_path = 'testset.csv'
    train, test = read_data(train_path, test_path)
    ktrain = pd.read_csv(train_path, header=None, names=['node1', 'node2'])
    ktest = pd.read_csv(test_path, header=None, names=['node1', 'node2'])
    g_train = build_graph(train)
    dw_train = build_graph(ktrain)

    walks = random_walks(dw_train, num_walks=8, walk_length=80)
    model = deepwalk_model(walks)
    predictions = generate_predictions(model, ktest)
    predicted_file_final(predictions, ktest, 'pred.csv')

if __name__ == '__main__':
    # Note if you want to run the main() file, please create a Results file under 47721300 file
    # Under Results file will be generated the results for kaggle
    # The normal result for project will under 47721300 file
    # with method name csv will be less good performance for the other models not the final One
    main()
    #main_final()
