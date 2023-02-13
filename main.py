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
    #########
    if DEBUG:
        print("sort_data_by_feature()", x)
    #########
    if len(D) <= 1:
        return D
    return D[D[:, x].argsort()]

# Run this subroutine for each numeric feature at each node of DT induction
def determine_candidate_numeric_splits(D): # set of training instances D, feature X
    #########
    if DEBUG:
        print("determine_candidate_numeric_splits()")
    #########
    C = []

    if len(D) <= 1:
        # Leaf node
        return C

    for Xi in [0, 1]: # assuming only 2 features
        D = sort_data_by_feature(D, Xi)
        #########
        if DEBUG:
            for x in D:
                print(x)
        #########
        for j in range(0, len(D)-1):
            if get_classification(D[j]) != get_classification(D[j+1]):
                C.append([Xi, D[j+1]])
    return C

def is_stopping_criteria_met(D, C):
    #########
    if DEBUG:
        print("is_stopping_criteria_met()")
        for c in C:
            print(c)
    #########

    stop = False

    # Covers all criteria:
    # The stopping criteria (for making a node into a leaf) are that
    # 1. the node is empty, or
    # -- Should never occur because should never split a set into itself and the empty set
    # -- Only occur as corner case of data is empty
    # 2. all splits have zero gain ratio (if the entropy of the split is non-zero), or
    # -- I do not know when this would occur when 3 does not occur, and 3 is covered
    # 3. the entropy of any candidates split is zero
    # -- Entropy is zero when all labels are the same, so when there are no candidate splits
    if len(D) <= 1 or C == []:
        stop = True

    # Never occurs: all splits have zero gain ratio (if the entropy of the split is non-zero)
    else:
        gain_ratios = []
        for c in range(0, len(C)):
            gain_ratio, S, split_feature, split_value = get_gain_ratios_of_C(C[c], D)
            gain_ratios.append(gain_ratio)
        stop = gain_ratios.count(0) == len(gain_ratios) # all splits have zero gain ratio
    #########
        if DEBUG:
            print(stop)

    if DEBUG:
        print(stop)
    #########
    return stop

def determine_classification(D):
    class_votes = [0, 0]
    for X in D:
        class_votes[get_classification(X)] = class_votes[get_classification(X)] + 1
    #########
    if DEBUG:
        print(class_votes)
    #########
    classification = 1 if class_votes[1] >= class_votes[0] else 0
    #########
    if DEBUG:
        print("classification:", classification)
        print("\n -------------------\n")
    #########
    return classification

def get_entropy(p, n, N):
    #########
    if DEBUG:
        print('get_entropy()')
    #########
    if n == 0 or p == 0:
        return 0
    else:
        return -(p / N)*math.log2(p / N)-(n / N)*math.log2(n / N)

def info_gain(D, S):
    if DEBUG:
        print('info_gain()')
    # Calculate entropy
    entropy = 0
    p = D[:, -1].tolist().count(1) # total classified as 1
    N = len(D) # total
    total_N = N
    n = N - p # total classified as 0
    #########
    if DEBUG:
        print('entropy', p, n, N)
    #########
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
    #########
    if DEBUG:
        print('true split', p, n, N)
    #########
    entropy_given_true = get_entropy(p, n, N)

    # false split
    entropy_given_false = 0
    N = len(S[1])
    false_N = N
    p = 0
    for x in S[1]:
        p = p + get_classification(x) # 0 or 1
    n = N - p
    #########
    if DEBUG:
        print('false split', p, n, N)
    #########
    entropy_given_false = get_entropy(p, n, N)

    # Remainder
    remainder = (true_N / total_N)*entropy_given_true + (false_N / total_N)*entropy_given_false

    # Calculate InfoGain
    info_gain = entropy - remainder
    return info_gain

def split_data(C, D):
    split_feature = C[0]
    split_value = C[1][split_feature]
    D = sort_data_by_feature(D, split_feature)
    S = [[],[]]
    for x in D:
        if x[split_feature] >= split_value:
            S[0].append(x)
        else:
            S[1].append(x)
    return S

def get_gain_ratios_of_C(C, D):
    #########
    if DEBUG:
        print('get_gain_ratios_of_C()')
    #########
    S = split_data(C, D)

    entropy_of_split = get_entropy(len(S[0]), len(S[1]), len(S[0]) + len(S[1]))
    #########
    if DEBUG:
        if entropy_of_split == 0:
            print(C, ' ENTROPY IS 0')
        print('S_c:',S)
    #########
    info_gain_c = info_gain(D, S)

    # skip if 0 split information
    gain_ratio = 0 if entropy_of_split == 0 \
        else info_gain_c / entropy_of_split

    return gain_ratio, S, C[0], C[1][C[0]]

def find_best_split(D, C):
    #########
    if DEBUG:
        print('find_best_split()')
        for c in C:
            print(c)
    #########
    max_gain_ratio = sys.float_info.min
    best_S = [[],[]]
    best_split_feature = -1
    best_split_value = -1
    for c in range(0, len(C)):
        #########
        if DEBUG:
            print('candidate:', C[c])
        #########
        gain_ratio, S_c, split_feature, split_value = get_gain_ratios_of_C(C[c], D)

        if (gain_ratio > max_gain_ratio):
            best_S = S_c
            max_gain_ratio = gain_ratio
            best_split_feature = split_feature
            best_split_value = split_value
    #########
        if DEBUG:
            print("gain_ratio of", c, 'is', gain_ratio)
            print('max_gain_ratio', max_gain_ratio)
            print()

    if DEBUG:
        print('best: ', split_feature, split_value)
    #########

    return best_S, best_split_value, best_split_feature

def make_subtree(D, root=False):
    # Section 2 question 3
    if root:
        C = determine_candidate_numeric_splits(D)
        for c in C:
            gain_ratio, S, split_feature, split_value = get_gain_ratios_of_C(c, D)
            info_gain_c = info_gain(D, S)
            print('candidate split: split on x%d >= %f' % (c[0], c[1][c[0]]), end='\t| ')
            if gain_ratio == 0:
                print('info_gain:', info_gain_c)
            else:
                print('gain_ratio:', gain_ratio)

    #########
    if DEBUG:
        print("making subtree with D\n", D, '', type(D),'\n:\n')
    #########
    C = determine_candidate_numeric_splits(D)
    if is_stopping_criteria_met(D, C):
        #########
        if DEBUG:
            print("reached a leaf node.")
        #########
        return TreeNode(is_leaf=True, classification=determine_classification(D))
    else:
        S, split_value, split_feature = find_best_split(D, C)
        #########
        if DEBUG:
            print("split_value, split_feature", split_value, split_feature)
            print("making 2 new subtrees with\n", S[0], '\n\n', S[1], '\n---\n')
        #########
        return TreeNode(left_branch=make_subtree(np.array(S[0])), right_branch=make_subtree(np.array(S[1])), split_value=split_value, split_feature=split_feature)

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
        D.append([float(line_split[0]), float(line_split[1]), int(float(line_split[2]))])

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

DEBUG = False

if __name__ == "__main__":
    D = np.array(file_input(sys.argv[1]))

    tree = make_subtree(D)

    print_tree(0, tree)
