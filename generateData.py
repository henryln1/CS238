import random
import collections
import csv

csv.register_dialect('pipes', delimiter='|')


csvFile = "data.csv"

def generateRandomState():
	domain = [-1,0,1]
	sleep = random.choice(domain)
	nutrition = random.choice(domain)
	activity = random.choice(domain)
	return (sleep, nutrition, activity)

actions = ["remindTruths", "rewardPunish", "activePlanning"]

def generateData(dataPoints):


	#csv = open(csvFile, 'w')

	#columnTitleRow = "state, action, nextState, reward\n"
	#csv.write(columnTitleRow)
	#Dialect.delimiter = "|"
	#with open(csvFile, 'w', newline = '') as csv:
	#	writer = csv.writer(csv, delimiter = '|')

	#with open('eggs.csv', 'w') as csvfile:
	 #   spamwriter = csv.writer(csvfile, delimiter='%', lineterminator = '\n')
	  #  spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
	   # spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

	#return
	file = open('data2.0.txt', 'w')

	for i in range(dataPoints):
		state = generateRandomState()
		action = random.choice(actions)
		nextState = generateRandomState()
		#reward = random.uniform(1,10)
		reward = 2 * nextState[0] + 2 * nextState[1] + 2 * nextState[2]
		file.write(str(state) + "|")
		file.write(str(action) + "|")
		file.write(str(nextState) + "|")
		file.write(str(reward))

		file.write('\n')
		#csv.write(row)


generateData(1000)

