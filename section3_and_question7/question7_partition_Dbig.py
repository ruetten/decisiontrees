import numpy as np

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

if __name__ == "__main__":
    Dbig = np.array(file_input('data/Dbig.txt'))
    np.random.shuffle(Dbig)
    D8192 = Dbig[:8192]
    Dtest = Dbig[8192:]

    D2048 = D8192[:2048]
    D512 = D2048[:512]
    D128 = D512[:128]
    D32 = D128[:32]

    np.savetxt('data/D8192.txt', D8192)
    np.savetxt('data/Dtest.txt', Dtest)
    np.savetxt('data/D2048.txt', D2048)
    np.savetxt('data/D512.txt', D512)
    np.savetxt('data/D128.txt', D128)
    np.savetxt('data/D32.txt', D32)
