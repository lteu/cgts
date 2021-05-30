"""

visualize heatmaps

change parameters:
inputFile for file location
topic for type of features 

call:
python3 showHeatmap.py

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import sys


def showRichHeatmap(loaded_data,topic):
	'''
	Heatmap format computed by heatmap.py
	'''
	metas = loaded_data['results']['metas']
	rowlen = len(metas[0])

	all_data = []
	for x in metas:
		row = []
		for i in range(rowlen):
			if i >= len(x):
				row.append(np.nan)
			else:
				if not isinstance(x[i][topic], float) and not isinstance(x[i][topic], int):
					row.append(np.nan)
					continue
				elif np.isnan(x[i][topic]):
					row.append(np.nan)
					continue
				v = float(x[i][topic])
				row.append(v)
		all_data.append(row)


	x_labels = [l[0] for l in loaded_data['results']['periods'][0]]

	return all_data,x_labels


def showSimpleHeatmap(loaded_data):
	'''
	Heatmap format computed by fastHeatmap.py
	'''

	data = loaded_data['results'][:-1]
	tlapse = loaded_data['time_lapse']
	x_labels = loaded_data['start_time_points'] if 'start_time_points' in loaded_data else []

	rowlen = len(data[0])

	all_data = []
	for x in data:
		row = []
		for i in range(rowlen):
			if i >= len(x):
				row.append(np.nan)
			else:
				row.append(x[i])
		all_data.append(row)

	return all_data,x_labels

def main(args):

	# inputFile = "out/rlt.json"
	inputFile = "out/tmp.json"

	with open(inputFile) as f:
		loaded_data=json.load(f)

	# check format
	if 'metas' in loaded_data['results']:
		# format - rich version
		topic = 'radius'
		# topic = 'first_degree_score'
		# topic = 'mod_score'
		all_data,x_labels = showRichHeatmap(loaded_data,topic)
	else:
		# simple version, no topic options
		all_data,x_labels = showSimpleHeatmap(loaded_data)
		

	# data to heatmap
	uniform_data = np.array(all_data)
	if len(x_labels) != 0:
		ax = sns.heatmap(uniform_data,xticklabels=x_labels)
		ax.set_xticklabels(x_labels, rotation=45)
	else:
		ax = sns.heatmap(uniform_data)	

	ax.invert_yaxis()
	plt.show()


if __name__ == '__main__':
	main(sys.argv[1:])

