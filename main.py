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

import numpy as np
import sys
import math

class TreeNode:
    def __init__(self, is_leaf=False, left_branch=None, right_branch=None, classification=-1, split_feature=-1, split_value=-1):
        self.is_leaf = is_leaf
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.classification = classification
        self.split_feature = split_feature
        self.split_value = split_value

def get_classification(example):
    return int(example[-1])

def sort_data_by_feature(D, x):
    #print("sort_data_by_feature()", x)
    if len(D) <= 1:
        return D
    return D[D[:, x].argsort()]

# Run this subroutine for each numeric feature at each node of DT induction
def determine_candidate_numeric_splits(D, feature_labels): # set of training instances D, feature X
    #print("determine_candidate_numeric_splits()")
    C = []

    if len(D) <= 1:
        #print("leaf node")
        return C

    for Xi in feature_labels:
        D = sort_data_by_feature(D, Xi)
        #for x in D:
        #    print(x)
        for j in range(0, len(D)-1):
            if get_classification(D[j]) != get_classification(D[j+1]):
                C.append([Xi, D[j+1]])
    return C

def is_stopping_criteria_met(D, C):
    #print("is_stopping_criteria_met()")
    #for c in C:
    #    print(c)
    met = len(D) <= 1 or C == []
    #print(met)
    return met

def determine_classification(D):
    class_votes = [0, 0]
    for X in D:
        class_votes[get_classification(X)] = class_votes[get_classification(X)] + 1
    classification = 0 if class_votes[0] >= class_votes[1] else 1
    #print("classification:", classification)
    #print("\n -------------------\n")
    return classification

def get_entropy(p, n, N):
    if n == 0 or p == 0:
        return 0
    else:
        return -(p / N)*math.log2(p / N)-(n / N)*math.log2(n / N)

def info_gain(D, S):
    # Calculate entropy
    entropy = 0
    p = D[:, -1].tolist().count(1) # total classified as 1
    N = len(D) # total
    total_N = N
    n = N - p # total classified as 0
    #print('entropy', p, n, N)
    entropy = get_entropy(p, n, N)

    # Calculate Remainder
    # true split
    entropy_given_true = 0
    N = len(S[0])
    true_N = N
    p = 0
    for x in S[0]:
        p = p + get_classification(x) # 0 or 1
    n = N - p
    #print("true split", p, n, N)
    entropy_given_true = get_entropy(p, n, N)

    # false split
    entropy_given_false = 0
    N = len(S[1])
    false_N = N
    p = 0
    for x in S[1]:
        p = p + get_classification(x) # 0 or 1
    n = N - p
    #print("false split", p, n, N)
    entropy_given_false = get_entropy(p, n, N)

    # Remainder
    remainder = (true_N / total_N)*entropy_given_true + (false_N / total_N)*entropy_given_false
    #print('remainder', remainder)

    # Calculate InfoGain
    info_gain = entropy - remainder
    return info_gain

def find_best_split(D, C):
    #print("find_best_split()")
    #for c in C:
    #    print(c)
    max_gain_ratio = sys.float_info.min
    best_S = [[],[]]
    best_c = -1
    best_split_feature = -1
    best_split_value = -1
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

        info_gain_c = info_gain(D, S_c)
        #print("info_gain of", c, 'is', info_gain_c)

        p = len(S_c[0])
        n = len(S_c[1])
        N = n + p
        entropy_of_split = get_entropy(p, n, N)

        gain_ratio = info_gain_c / entropy_of_split

        if (gain_ratio > max_gain_ratio):
            best_S = S_c
            max_gain_ratio = gain_ratio
            best_split_feature = split_feature
            best_split_value = split_value
            best_c = c

        #print("gain_ratio of", c, 'is', gain_ratio)
        #print('max_gain_ratio', max_gain_ratio)

        #print()
    #print("best split was", best_c)
    return best_S, best_split_value, best_split_feature

def make_subtree(D, feature_labels):
    #print("making subtree with D\n", D, '', type(D),'\n:\n')
    C = determine_candidate_numeric_splits(D, feature_labels)
    if is_stopping_criteria_met(D, C):
        #print("reached a leaf node.")
        return TreeNode(is_leaf=True, classification=determine_classification(D))
    else:
        S, split_value, split_feature = find_best_split(D, C)
        #print("split_value, split_feature", split_value, split_feature)
        #print("making 2 new subtrees with\n", S[0], '\n\n', S[1], '\n---\n')
        return TreeNode(left_branch=make_subtree(np.array(S[0]), feature_labels), right_branch=make_subtree(np.array(S[1]), feature_labels), split_value=split_value, split_feature=split_feature)

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

def print_tree(level, tree):
    if tree.is_leaf:
        for i in range(0, level):
            print("   |",end='')
        print("  ",end='')
        print("└──",end='')
        print(tree.classification)
    else:
        for i in range(0, level):
            print("   |",end='')
        print("  ",end='')
        print('x'+str(tree.split_feature),'>=',tree.split_value)

        for i in range(0, level):
            print("   |",end='')
        print("  ",end='')
        print("└──",end='')
        print()
        print_tree(level+1, tree.left_branch)

        for i in range(0, level):
            print("   |",end='')
        print("  ",end='')
        print("└──",end='')
        print()
        print_tree(level+1, tree.right_branch)

################################################################################

if __name__ == "__main__":
    D = file_input('data/Dbig.txt')
    D = np.array(D)

    # for x in sort_data_by_feature(D, 1):
    #     print(x)

    feature_labels = [0, 1]
    tree = make_subtree(D, feature_labels)

    print_tree(0, tree)

    #parent_examples = []
    #decision_tree_learning(examples, feature_labels, parent_examples)
