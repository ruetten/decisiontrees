from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sys

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

tree = DecisionTreeClassifier(random_state=0)
Dtrain = np.array(file_input(sys.argv[1]))
Dtest = np.array(file_input(sys.argv[2]))

Xtrain = Dtrain[:, :-1]
ytrain = Dtrain[:, -1]

Xtest = Dtest[:, :-1]
ytest = Dtest[:, -1]

tree.fit(Xtrain, ytrain)
print('n leaves:',tree.get_n_leaves())

score = tree.score(Xtest, ytest)
print('err:', 1-score)

print(len(Dtrain),1-score)
