import numpy as np
import pandas as pd
from collections import Counter

# Entropy & Info Gain
def get_entropy_of_dataset(data: np.ndarray) -> float:
    target_col = data[:, -1]
    classes, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    total_rows = data.shape[0]
    avg_info = 0.0
    for value in np.unique(data[:, attribute]):
        subset = data[data[:, attribute] == value]
        weight = subset.shape[0] / total_rows
        avg_info += weight * get_entropy_of_dataset(subset)
    return avg_info

def get_information_gain(data: np.ndarray, attribute: int) -> float:
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    return round(dataset_entropy - avg_info, 4)

def get_selected_attribute(data: np.ndarray) -> tuple:
    n_attributes = data.shape[1] - 1
    gain_dict = {attr: get_information_gain(data, attr) for attr in range(n_attributes)}
    selected_attr = max(gain_dict, key=gain_dict.get)
    return gain_dict, selected_attr


# Decision Tree
class DecisionTreeNode:
    def __init__(self, attribute=None, value=None, label=None):
        self.attribute = attribute
        self.value = value
        self.label = label
        self.children = {}

    def is_leaf(self):
        return self.label is not None

def build_tree(data: np.ndarray) -> DecisionTreeNode:
    target_col = data[:, -1]
    unique_classes = np.unique(target_col)

    if len(unique_classes) == 1:
        return DecisionTreeNode(label=unique_classes[0])

    if data.shape[1] == 1:
        majority = Counter(target_col).most_common(1)[0][0]
        return DecisionTreeNode(label=majority)

    _, best_attr = get_selected_attribute(data)
    node = DecisionTreeNode(attribute=best_attr)

    for value in np.unique(data[:, best_attr]):
        subset = data[data[:, best_attr] == value]
        if subset.shape[0] == 0:
            majority = Counter(target_col).most_common(1)[0][0]
            node.children[value] = DecisionTreeNode(label=majority)
        else:
            reduced_subset = np.delete(subset, best_attr, axis=1)
            node.children[value] = build_tree(reduced_subset)

    return node


# Prediction & Evaluation
def predict(tree: DecisionTreeNode, sample: np.ndarray) -> str:
    if tree.is_leaf():
        return tree.label

    attr_val = sample[tree.attribute]
    child = tree.children.get(attr_val)

    if child is None:  # unseen valeu → fallback majorty
        return Counter([c.label for c in tree.children.values() if c.is_leaf()]).most_common(1)[0][0]

    reduced_sample = np.delete(sample, tree.attribute)
    return predict(child, reduced_sample)

def evaluate(tree: DecisionTreeNode, data: np.ndarray):
    y_true = data[:, -1]
    y_pred = [predict(tree, row[:-1]) for row in data]

    accuracy = np.mean(y_true == y_pred)
    metrics = {}
    for cls in np.unique(y_true):
        tp = np.sum((np.array(y_pred) == cls) & (y_true == cls))
        fp = np.sum((np.array(y_pred) == cls) & (y_true != cls))
        fn = np.sum((np.array(y_pred) != cls) & (y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}
    return accuracy, metrics


# Tree complexity
def tree_depth(node: DecisionTreeNode) -> int:
    if node.is_leaf():
        return 1
    return 1 + max(tree_depth(child) for child in node.children.values())

def node_count(node: DecisionTreeNode) -> int:
    if node.is_leaf():
        return 1
    return 1 + sum(node_count(child) for child in node.children.values())


# Run on datasets
if __name__ == "__main__":
    datasets = ["mushrooms.csv", "tictactoe.csv", "Nursery.csv"]

    for fname in datasets:
        print("\n==============================")
        print(f" Daataset: {fname}")  # typo added
        print("==============================")
        df = pd.read_csv(fname)
        data = df.values

        tree = build_tree(data)
        acc, metrics = evaluate(tree, data)
        depth = tree_depth(tree)
        nodes = node_count(tree)

        print(f"Acuracy: {acc:.4f}")  # typo added
        print(f"Tree Depth: {depth}")
        print(f"Number of Nodes: {nodes}")
        print("Metircs per class:")  # typo added
        for cls, m in metrics.items():
            print(f"  Class {cls}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1']:.3f}")
