import sys
import math
import numpy as np 
import csv
import pandas
import random
import collections

allActions = ['negativeReinforcement', 'activePlanning', 'positiveReinforcement']
stateList = set()

def findBestAction(Q, state):
	maxValue = float('-infinity')
	bestAction = None
	for key in Q:
		currentS, action = key
		if (currentS == state and Q[(currentS, action)] >= maxValue):
			maxValue = Q[(currentS, action)]
			bestAction = action
	if (maxValue == float('-infinity')):
		maxValue = 0
		bestAction = random.choice(allActions)
	return bestAction


def write_policy(Q, filename, num_states, state_shape):
	policy = []
	with open(filename, 'w') as f:
		#for s in range(num_states):
		for s in stateList:
			best_action = findBestAction(Q, s)
			f.write('State '+ str(s) + ' has best action ' + str(best_action) + '\n')
			policy.append(best_action)
	#if num_states < 200:
	 #   print(np.reshape(policy, state_shape))

def compute(infile, outfile):
	#load in the file
	reader = csv.reader(open(infile), delimiter = ',')
	inputInformation = pandas.read_csv(infile)
	lines = list(reader)
	D = lines[1:]

	#create numpy matrix where each row is a data point
	#d_matrix = np.array(D).astype(tuple)
	#d_matrix[:, 0] -= 1 # s
	#d_matrix[:, 1] -= 1 # a
	#d_matrix[:, 3] -= 1 # sp

	csvList = []

	#uses pandas to create a list with all the information)
	for index, row in inputInformation.iterrows(): #goes through the csv
		state =row['s']
		action = row['a']        
		reward = row['r']
		nextState = row['sp']
		csvList.append([state, action, reward, nextState])
		if (state not in stateList):
			stateList.add(state)


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
	elif infile == 'fixedData.csv':
		num_actions = 3
		num_states = 27
		num_iterations = 500
		state_shape = (0,0)
		gamma = 1
	else: #infile = "large.csv"
		num_actions = 125
		num_states = 10101010
		num_iterations = 2000000
		state_shape = (0, 0)
		gamma = 0.95

	alpha = 0.01

	def findMaxQ(Q, state):
		maxValue = float('-infinity')
		for key in Q:
			currentS, action = key
			if (currentS == state and Q[(currentS, action)] >= maxValue):
				maxValue = Q[(currentS, action)]
		if (maxValue == float('-infinity')):
			maxValue = 0
		return maxValue

	#Q = np.zeros((num_states, num_actions))
	Q = collections.defaultdict(float)
	for i in range(num_iterations):
		if (i % 100000 == 0): print(i)
		#sample = d_matrix[np.random.randint(0, d_matrix.shape[0])]
		sample = random.choice(csvList)
		#print sample
		[s, a, r, sp] = sample

		#Q-learning form
		#compute Q_samp and difference
		#Q_samp = r + gamma * findMaxQ(Q, sp)
		#difference = Q_samp - Q[(s, a)]

		#SARSA variation with epsilon greedy exploration policy
		epsilon = 0.3
		prob = random.uniform(0.0,1.0)
		if (prob <= epsilon):
			randomAction = random.choice(allActions)
			SARSA_sample = r + gamma * Q[(sp, randomAction)]
			difference = SARSA_sample - Q[(s,a)]
			#update Q
			Q[(s, a)] += alpha * difference
		else:  #regular Q-learning
			Q_samp = r + gamma * findMaxQ(Q, sp)
			difference = Q_samp - Q[(s, a)]
			Q[(s, a)] += alpha * difference



	write_policy(Q, outfile, num_states, state_shape)

def main():
	if len(sys.argv) != 3:
		raise Exception("usage: python Qlearning.py <infile>.csv <outfile>.policy")

	inputfilename = sys.argv[1]
	outputfilename = sys.argv[2]
	compute(inputfilename, outputfilename)

if __name__ == '__main__':
	main()
