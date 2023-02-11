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

class TreeNode:
    def __init__(self, root):
        root = None

    def add_branch(v, subtree):
        pass

def plurality_value(parent_examples):
    pass

def most_important_attribute(attributes, examples):
    pass

def values_of_A(A):
    pass

def get_exs(examples, A, v):
    pass

def decision_tree_learning(examples, attributes, parent_examples):
    # Some stopping criteria
    if len(examples) == 0:
        return plurality_value(parent_examples)
    elif all_examples_have_same_classification(examples):
        #TODO return the classification
        return
    elif len(attributes) == 0:
        return plurality_value(examples)
    # else make a new subtree
    else:
        A = most_important_attribute(attributes, examples)
        tree = TreeNode(A)
        for v in values_of_A(A):
            exs = get_exs(examples, A, v)
            attributes.remove(A) # TODO test
            subtree = decision_tree_learning(exs, attributes, examples)
            tree.add_branch(v, subtree)
        return tree

################################################################################
# CS760 Class notes code
################################################################################

# Run this subroutine for each numeric feature at each node of DT induction
def determine_candidate_numeric_splits(D, Xi) # set of training instances D, feature X
    C = [] # initialize set of candidate splits for feature Xi
    # sort the dataset using vj as the key for each data point
    # for each pair of adjacent vj, vj+1 in the sorted order
    #   if the corresponding class labels are different
    #       add candidate split Xi <= (vj + vjp1)/2 to C
    return C

def make_subtree(training_data):
    C = determine_candidate_splits(D, X[i])
    if is_stopping_criteria_met():
        N = make_leaf_node()
        determine_class_label_for(N)
    else:
        N = make_internal_node()
        S = FindBestSplit(D, C)
        for k in range(0,len(S)):
            D = subset_of_training_data_group_k(S[k])
            N[k] = MakeSubtree(D[k])
    return subtree_rooted_at_N()

################################################################################

def file_input(filename):
    global X
    global y
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
        X.append([float(line_split[0]), float(line_split[1])])
        y.append(int(line_split[2]))

    file.close()


################################################################################

X = []
y = []
if __name__ == "__main__":
    file_input('data/D1.txt')
    print(X)

    examples = []
    attributes = []
    parent_examples = []
    decision_tree_learning(examples, attributes, parent_examples)
