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
import matplotlib.pyplot as plt

###
# Class for each node of the decision tree
# is_leaf - boolean for if is a leaf node or not
# left_branch - 'then' side of the decision tree
# right_branch - 'else' side of the decision tree
# classification - if is_leaf, then store classification of is class 0 or 1
# split_feature - the feature to split on
# split_value - the value to compare split_feature to
# split_feature and split_value - at non-leaf nodes, these two values represent
#   the rule that splits the data at this node in the format of:
#   x_(split_feature) >= split_value
class TreeNode:
    def __init__(self, is_leaf=False, left_branch=None, right_branch=None, classification=-1, split_feature=-1, split_value=-1):
        self.is_leaf = is_leaf
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.classification = classification
        self.split_feature = split_feature
        self.split_value = split_value

# Simple function, but makes code more readable
# returns classification of the example
def get_classification(example):
    return int(example[-1])

# Sort data D by feature x (0 or 1 whether it is feature x1 or x2, respectively)
def sort_data_by_feature(D, x):
    #########
    if DEBUG:
        print("sort_data_by_feature()", x)
    #########
    if len(D) <= 1:
        return D
    return D[D[:, x].argsort()]

# Determine the candidate splits for numeric datasets
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
        for j in range(0, len(D)-1):
            if get_classification(D[j]) != get_classification(D[j+1]):
                C.append([Xi, D[j+1]])
                # Handles the case that the sorted dataset could have the same values
                if D[j, Xi] == D[j+1, Xi]:
                    for k in range(j+1, len(D)-1):
                        if D[k, Xi] == D[k+1, Xi]:
                            C.append([Xi, D[k+1]])
                            continue
    return C

# Checks if tree has met stopping criteria to create a leaf node
# D - current dataset
# C - candidate splits
def is_stopping_criteria_met(D, C):
    #########
    if DEBUG:
        print("is_stopping_criteria_met()")
        for c in C:
            print(c)
    #########

    stop = False

    # Just checking that there are no candidate splits covers all criteria:
    # The stopping criteria (for making a node into a leaf) are that
    # 1. the node is empty, or
    # -- Should never occur because should never split a set into itself and the empty set
    # -- Only occur as corner case of data is empty
    # 2. all splits have zero gain ratio (if the entropy of the split is non-zero), or
    # -- I do not know when this would occur when 3 does not occur, and 3 is covered
    # 3. the entropy of all candidates split is zero
    if C == []:
        stop = True

    #########
        if DEBUG:
            print(stop)

    if DEBUG:
        print(stop)
    #########
    return stop

# Determine the classification of the the dataset D
# This method is used to create a leaf node
# returns classification, which is either 0 or 1
# determined by class majority of D
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

# Equation for entropy
# p - positive cases out of N
# n - negatice cases out of N
# N - total number of values to calculate probability in entropy equation
def get_entropy(p, n, N):
    #########
    if DEBUG:
        print('get_entropy()')
    #########
    if n == 0 or p == 0:
        return 0
    else:
        return -(p / N)*math.log2(p / N)-(n / N)*math.log2(n / N)

# Calculates information gain
# D - original dataset
# S - split dataset
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

# Splits data D at split candidate C
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

# Calculate gain ratio of candidate split
# C - the candidate split
# D - the dataset to split
def get_gain_ratio_of_C(C, D):
    #########
    if DEBUG:
        print('get_gain_ratio_of_C()')
    #########
    S = split_data(C, D)

    entropy_of_split = get_entropy(len(S[0]), len(S[1]), len(S[0]) + len(S[1]))
    #########
    if DEBUG:
        if entropy_of_split == 0:
            print(C, ' ENTROPY IS 0')
    #########
    # skip if 0 split information
    if entropy_of_split == 0:
        return 0, S, C[0], C[1][C[0]]

    info_gain_c = info_gain(D, S)

    gain_ratio = info_gain_c / entropy_of_split

    return gain_ratio, S, C[0], C[1][C[0]]

# Find the best split best_S out of all candidate splits C on dataset D
def find_best_split(D, C):
    #########
    if DEBUG:
        print('find_best_split()')
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
        gain_ratio, S_c, split_feature, split_value = get_gain_ratio_of_C(C[c], D)

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

###
# The main equation for building the decision tree
# D - dataset to build tree for
# root - Only used for Section 2 question 3, True if first node in the tree
def make_subtree(D, root=False):
    # Only used for Section 2 question 3
    if root:
        C = determine_candidate_numeric_splits(D)
        for c in C:
            gain_ratio, S, split_feature, split_value = get_gain_ratio_of_C(c, D)
            info_gain_c = info_gain(D, S)
            print('candidate split: split on x%d >= %f' % (c[0]+1, c[1][c[0]]), end='\t| ')
            if gain_ratio == 0:
                print('info_gain:', info_gain_c)
            else:
                print('gain_ratio:', gain_ratio)

    #########
    if DEBUG:
        print("making subtree with D\n", D, '', type(D),'\n:\n')
    #########

    ### MAIN ALGORITHM STARTS HERE
    # Determine candidate splits
    # C is an array of candidate splits,
    # and each c in C is in the format of:
    # [split_feature, D[i]]
    # split_feature being 0 if feature x1 or 1 if feature x2
    # D[i] being the row example within D where the split will occur
    C = determine_candidate_numeric_splits(D)
    # If stopping criteria met, create a tree node; make subtrees otherwise
    if is_stopping_criteria_met(D, C):
        #########
        if DEBUG:
            print("reached a leaf node.")
        #########
        return TreeNode(is_leaf=True, classification=determine_classification(D))
    else:
        # S is the split dataset in the format of [D[:{split_location}], D[{split_location}:]]
        # split rule in tree looks like split_feature >= split_value
        S, split_value, split_feature = find_best_split(D, C)
        #########
        if DEBUG:
            print("split_value, split_feature", split_value, split_feature)
            print("making 2 new subtrees with\n", S[0], '\n\n', S[1], '\n---\n')
        #########
        # If the split attempted resulted in D -> S = [D, []], then we are actually at a leaf node
        if len(S[1]) == 0:
            return TreeNode(is_leaf=True, classification=determine_classification(D))
        # Create subtrees for left and right branches
        return TreeNode(left_branch=make_subtree(np.array(S[0])), right_branch=make_subtree(np.array(S[1])), split_value=split_value, split_feature=split_feature)

#######################Post-tree-building testing###############################

# Traverse the tree with datapoint x to get the class of x
def get_class_of_x_from_tree(tree, x):
    if (tree.is_leaf):
        return tree.classification
    else:
        if x[tree.split_feature] >= tree.split_value:
            return get_class_of_x_from_tree(tree.left_branch, x)
        else:
            return get_class_of_x_from_tree(tree.right_branch, x)

# Test tree with dataset D
def test_tree(tree, D):
    N = len(D)
    num_correct = 0
    for x in D:
        if x[-1] == get_class_of_x_from_tree(tree, x):
            num_correct = num_correct + 1
    print(num_correct, '/', N, '=', num_correct / N,'err=', 1 - num_correct / N)

# Count the number of trees in tree
def count_nodes_in_tree(tree):
    if tree.is_leaf:
        return 1
    else:
        return 1 + count_nodes_in_tree(tree.left_branch) + count_nodes_in_tree(tree.right_branch)

#################################File I/O#######################################

# Input D.txt file for dataset
# filename - name of the file, usually D{?}.txt for this assignemnt
# returns dataset D from file
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

# Print a text visualization for the tree
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
        print('x'+str(tree.split_feature+1),'>=',tree.split_value)

        for i in range(0, level):
            print("   |",end='')
        print("  ",end='')
        print("└then ",end='')
        print()
        print_tree(level+1, tree.left_branch)

        for i in range(0, level):
            print("   |",end='')
        print("  ",end='')
        print("└else ",end='')
        print()
        print_tree(level+1, tree.right_branch)

# Create a scatter plot of dataset D into a file with name filename
# produces an output .png in the output folder
def scatter_plot_points(D, filename):
    x_class_0 = D[D[:, 2] == 0, :]
    x_class_1 = D[D[:, 2] == 1, :]
    plt.scatter(x_class_0[:, 0], x_class_0[:, 1], color = 'red')
    plt.scatter(x_class_1[:, 0], x_class_1[:, 1], color = 'blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(['0', '1'])
    plt.title(filename)
    plt.savefig('output/'+filename+'.png')

# Draws the decision boundary for the scatterplot of points
# Works by creating A LOT of smaller points to just fill in the boundary space,
# then plotting the scatterplot on top
# D - dataset to plot
# tree - decision tree to draw the boundaries of
# filename - name of the file to produce an output to
def draw_scatter_plot_with_boundary(D, tree, filename):
    x_class_0 = D[D[:, 2] == 0, :]
    x_class_1 = D[D[:, 2] == 1, :]

    N = 200
    boundary = [[],[]]
    for x0 in np.linspace(min(D[:, 0]), max(D[:, 0]), N):
        for x1 in np.linspace(min(D[:, 1]), max(D[:, 1]), N):
            boundary[get_class_of_x_from_tree(tree, [x0, x1])].append([x0, x1])

    boundary_for_class0 = np.array(boundary[0]).T
    boundary_for_class1 = np.array(boundary[1]).T

    plt.scatter(boundary_for_class0[0], boundary_for_class0[1], color = 'gray', s=2)
    plt.scatter(boundary_for_class1[0], boundary_for_class1[1], color = 'black', s=2)
    plt.scatter(x_class_0[:, 0], x_class_0[:, 1], color = 'red', s=5)
    plt.scatter(x_class_1[:, 0], x_class_1[:, 1], color = 'blue', s=5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(['0', '1'])
    plt.title(filename)
    plt.savefig('output/'+filename+'.png')

################################################################################

# Set to true to turn on all DEBUG print statements
DEBUG = False

# My main method that I commented out and maniputed a lot in order to produce what I needed
if __name__ == "__main__":
    filename = sys.argv[1] # take a command line argument to pull in filename for dataset D
    D = np.array(file_input(filename))

    tree = make_subtree(D)
    print_tree(0, tree)
    #scatter_plot_points(D, sys.argv[1].split('/')[-1])
    #draw_scatter_plot_with_boundary(D, tree, sys.argv[1].split('/')[-1])

    print('num nodes =', count_nodes_in_tree(tree))
    #test_tree(tree, np.array(file_input('data/Dtest.txt')))
    test_tree(tree, D)
