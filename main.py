################################################################################
################################################################################
# Laik Ruetten
# lruetten
# CS 760 Machine Learning
# Spring 2023
# Dr. Kirthevasan Kandasamy
# UW-Madison
#
# Homework 2: Decision Trees
#
################################################################################
################################################################################

################################################################################
# Textbook version of the decision tree
################################################################################
import numpy as np
import sys
import math

class TreeNode:
    def __init__(self, is_leaf=False, left_branch=None, right_branch=None, classification=-1):
        self.is_leaf = is_leaf
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.classification = classification

def plurality_value(examples):
    print("plurality_value()")

def get_classification(example):
    return int(example[-1])

def all_examples_have_same_classification(examples):
    print("all_examples_have_same_classification()")
    # if all are the same, all are the same as the first example
    classification = get_classification(examples[0])
    for ex in examples:
        if get_classification(ex) != classification:
            print("not the same")
            return False
    return True

def most_important_attribute(attributes, examples):
    print("most_important_attribute()")
    A = -1
    maxA = sys.float_info.min
    for a in attributes:
        importance = importance(a, examples)
        if (importance > maxA):
            maxA = importance
            A = a
    if A == -1:
        return attributes[0] ## wha???
    return A

def values_of_A(A):
    print("values_of_A()")
    return [0, 1]

def get_exs(examples, A, v):
    pass

def decision_tree_learning(examples, attributes, parent_examples):
    # Some stopping criteria
    if len(examples) == 0:
        return plurality_value(parent_examples)
    elif all_examples_have_same_classification(examples):
        print("all the same")
        return examples[2] # index 2 of D is y
    elif len(attributes) == 0:
        return plurality_value(examples)
    # else make a new subtree
    else:
        A = most_important_attribute(attributes, examples)
        tree = TreeNode(A)
        for v in values_of_A(A):
            exs = get_exs(examples, A, v)
            if A in attributes:
                attributes.remove(A)
            subtree = decision_tree_learning(exs, attributes, examples)
            tree.add_branch(v, subtree)
        return tree

################################################################################
# CS760 Class notes code
################################################################################
def sort_data_by_feature(D, x):
    print("sort_data_by_feature()", x)
    if len(D) <= 1:
        return D
    return D[D[:, x].argsort()]

# Run this subroutine for each numeric feature at each node of DT induction
def determine_candidate_numeric_splits(D, feature_labels): # set of training instances D, feature X
    print("determine_candidate_numeric_splits()")
    C = []
    for Xi in feature_labels:
        D = sort_data_by_feature(D, Xi)
        for j in range(0, len(D)-1):
            if get_classification(D[j]) != get_classification(D[j+1]):
                C.append([Xi, D[j+1]])
    return C

def is_stopping_criteria_met(D, C):
    print("is_stopping_criteria_met()")
    met = len(D) <= 1
    print(met)
    return met

def determine_classification(D):
    class_votes = [0, 0]
    for X in D:
        print(get_classification(X))
        class_votes[get_classification(X)] = class_votes[get_classification(X)] + 1
    return 0 if class_votes[0] >= class_votes[1] else 1

def info_gain(D, positive_split):
    # Calculate entropy
    entropy = 0
    p = np.sum(D[:, -1]) # total classified as 1
    N = len(D) # total
    n = N - p # total classified as 0
    if n > 0:
        entropy = -(p / N)*math.log2(p / N)-(n / N)*math.log2(n / N)

    # Calculate Remainder
    remainder = 0
    N = len(positive_split)
    p = 0
    for x in positive_split:
        p = p + get_classification(x) # 0 or 1
    n = N - p
    if n > 0:
        remainder = -(p / N)*math.log2(p / N)-(n / N)*math.log2(n / N)

    # Calculate InfoGain
    info_gain = entropy - remainder
    return info_gain

def find_best_split(D, C):
    print("find_best_split()", C)
    max_info_gain = sys.float_info.min
    best_S = [[],[]]
    best_c = -1
    for c in range(0, len(C)):
        # Perform the split
        split_feature = C[c][0]
        split_value = C[c][1][split_feature]
        D = sort_data_by_feature(D, split_feature)
        S_c = [[],[]]
        for x in D:
            if x[split_feature] >= split_value:
                S_c[0].append(x)
            else:
                S_c[1].append(x)

        info_gain_c = info_gain(D, S_c[0])
        if (info_gain_c > max_info_gain):
             best_S = S_c
             max_info_gain = info_gain_c
             best_c = -1
    print("best split was", best_c)
    return best_S

def make_subtree(D, feature_labels):
    C = determine_candidate_numeric_splits(D, feature_labels)
    if is_stopping_criteria_met(D, C):
        print("reached a leaf node")
        return TreeNode(is_leaf=True, classification=determine_classification(D))
    else:
        S = find_best_split(D, C)
        print("making 2 new subtrees with", S)
        print()
        return TreeNode(left_branch=make_subtree(S[0], feature_labels), right_branch=make_subtree(S[1], feature_labels))

################################################################################

def file_input(filename):
    D = []
    file = open(filename)
    count = 0
    while True:
        count += 1

        # Get next line from file
        line = file.readline()
        # if line is empty
        # end of file is reached
        if not line:
            break

        line_split = line.split()
        D.append([float(line_split[0]), float(line_split[1]), int(line_split[2])])

    file.close()
    return D


################################################################################

if __name__ == "__main__":
    D = file_input('data/Dmytest.txt')
    D = np.array(D)

    feature_labels = [0, 1]
    make_subtree(D, feature_labels)

    #parent_examples = []
    #decision_tree_learning(examples, feature_labels, parent_examples)
