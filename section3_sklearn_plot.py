import numpy as np
import matplotlib.pyplot as plt

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
        D.append([int(line_split[0]), float(line_split[1])])

    file.close()
    return D

D = file_input('section2_sklearn_n_vs_err.txt')
D = np.array(D)
D = np.transpose(D)

plt.plot(D[0], 100*D[1])
plt.title('n vs err')
plt.xlabel('n')
plt.ylabel('err_n (%)')
plt.savefig('n_vs_err.png')
