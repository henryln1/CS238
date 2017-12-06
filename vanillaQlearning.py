import sys
import math
import numpy as np 
import csv

def write_policy(Q, filename, num_states, state_shape):
    policy = []
    with open(filename, 'w') as f:
        for s in range(num_states):
            best_action = np.argmax(Q[s, :]) + 1
            f.write(str(best_action) + '\n')
            policy.append(best_action)
    if num_states < 200:
        print(np.reshape(policy, state_shape))

def compute(infile, outfile):
    #load in the file
    reader = csv.reader(open(infile), delimiter = ',')
    lines = list(reader)
    D = lines[1:]

    #create numpy matrix where each row is a data point
    d_matrix = np.array(D).astype(int)
    d_matrix[:, 0] -= 1 # s
    d_matrix[:, 1] -= 1 # a
    d_matrix[:, 3] -= 1 # sp

    if infile == 'small.csv':
        #num of actions is 4
        num_actions = 4
        num_states = 100
        num_iterations = 200000
        state_shape = (10, 10)
        gamma = 0.95
    elif infile == 'medium.csv':
        num_actions = 7
        num_states = 50000
        num_iterations = 200000
        state_shape = (100, 500)
        gamma = 1
    else: #infile = "large.csv"
        num_actions = 125
        num_states = 10101010
        num_iterations = 2000000
        state_shape = (0, 0)
        gamma = 0.95

    alpha = 0.01
    Q = np.zeros((num_states, num_actions))
    for i in range(num_iterations):
        if (i % 100000 == 0) print(i)
        sample = d_matrix[np.random.randint(0, d_matrix.shape[0])]
        [s, a, r, sp] = sample

        #compute Q_samp and difference
        Q_samp = r + gamma * np.max(Q[sp, :])
        difference = Q_samp - Q[s, a]

        #update Q
        Q[s, a] += alpha * difference

    write_policy(Q, outfile, num_states, state_shape)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.policy")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == '__main__':
    main()
