import numpy as np

from sklearn import datasets
from random import randrange

iris = datasets.load_iris()
X = iris.data
Y = iris.target
data = np.concatenate((X, Y.reshape((1, 150)).T), axis=1)


def gini_index(groups, classes):
    # all observations.
    n_obs = sum([len(group) for group in groups])
    gini = 0
    for group in groups:
        # nb observation in each group
        size = float(len(group))
        if size == 0:
            continue
        score = 0
        for val in classes:
            prob = [row[-1] for row in group].count(val)/size
            score += prob * prob
        gini += (1 - score) * (size / n_obs)
    return gini


def split_node(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return [left, right]


def get_split(dataset, n_features):
    classes = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    # loop on all features
    while len(features) < n_features:
        # get randomly a feature index and add it in the features list:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)

    for index in features:
        for row in dataset:
            groups = split_node(index, row[index], dataset)
            gini = gini_index(groups, classes)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    list_of_class = [row[-1] for row in group]
    # take the majority of group:
    return max(list_of_class, key=list_of_class.count)


def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    # if one of the two groups is none, make it to a terminal.
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return None

    # check for max depth.
    if depth >= max_depth:
        node['left'] = to_terminal(left)
        node['right'] = to_terminal(right)
        return None

    # go deeper in the left.
    # check if the left node has enough observations.
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


def predict(node, row):
    if row[node['index']] < node["value"]:
        # check if it's a terminal:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def sampling(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def vote_majority(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = sampling(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [vote_majority(trees, row) for row in test]
    return predictions










