"""
incremental heatmap computation

Mandatory:
--csv
--out


python3 fastHeatmap.py --csv ../data/sample.csv  --out out/rlt.json

python3 fastHeatmap.py --csv ../data/sample.csv --winSizeStep 10 --out out/rlt.json


"""


import sys
import os 
import pandas as pd 
import numpy as np 
import random
import os
import json
from heatmap import parseInputs,firstDegreeScore
import collections
import networkx as nx

root_arr = os.path.realpath(__file__).split('/')[:-2]
root = '/'.join(root_arr) 
sys.path.append(root+"/lib")

from mst import subFrame,loadCsv,corrToDistance,minimum_spanning_tree
from mstweb import elaborateData
import math
import time

# include binaries
from increm import getBaseMatrixAndDerivatesContd,getFirstRow,getFollowingRow

def analysis(edges):
	'''
	Graph to graph features
	Only degree distribution and first degree score are considered
	to personalize the analysis target, please modify this function.
	'''
	G=nx.Graph()

	for edge in edges:
		x = edge[0]
		y = edge[1]
		G.add_edge(x,y)

	degree_sequence = sorted([d for n, d in G.degree()], reverse=False)  # please use the latest networkx library, otherwise this would be problematic

	degreeCount = collections.Counter(degree_sequence)
	degreeDist = list(degreeCount.items())

	firstDgScore = firstDegreeScore(degreeDist)
	return degreeDist,round(firstDgScore,4)


def getValueList(result):
	'''
	Correlation matrix to features of MSTs
	'''
	value_array = []
	# msts = []
	for item in result:
		matrix_unsymmetric = corrToDistance(pd.DataFrame(item))
		Tcsr = minimum_spanning_tree(matrix_unsymmetric)
		arr= Tcsr.toarray().astype(float)
		edges = []
		for x in range(len(arr)):
			for y in range(len(arr[x])):
				if float(arr[x][y]) != 0.0:
					edges.append([x,y,round(arr[x][y],2)]) 

		degreeDist,val = analysis(edges)
		value_array.append(val)
		# msts.append(edges)

	return value_array


def incremental_cmp(names,df,steplen = 3):
	'''
	incremental heatmap computation using compiled binary code
	'''

	number_rows = len(df)

	#get column
	cols = df.columns
	n_cols = len(cols)

	T = len(df)
	K = len(df.columns)

	range_granularity = int(math.floor(T/steplen))

	final = []

	start_time_points = []

	# --- Efficient method ---

	# base components
	list_n = np.zeros((n_cols, n_cols),dtype=int)
	list_sum_prod = np.zeros((n_cols, n_cols))
	list_x_s_sum = np.zeros((n_cols, n_cols))
	list_x_sum_s = np.zeros((n_cols, n_cols))
	list_y_s_sum = np.zeros((n_cols, n_cols))
	list_y_sum_s = np.zeros((n_cols, n_cols))

	# running components
	in_list_n = np.zeros((n_cols, n_cols),dtype=int)
	in_list_sum_prod = np.zeros((n_cols, n_cols))
	in_list_x_s_sum = np.zeros((n_cols, n_cols))
	in_list_x_sum_s = np.zeros((n_cols, n_cols))
	in_list_y_s_sum = np.zeros((n_cols, n_cols))
	in_list_y_sum_s = np.zeros((n_cols, n_cols))

	numeric_df = df
	mat = numeric_df.to_numpy(dtype=float, na_value=np.nan, copy=False)

	# get base components
	list_n,list_sum_prod, list_x_s_sum, list_x_sum_s, list_y_s_sum, list_y_sum_s = getFirstRow(mat,steplen,math.floor(len(df)/steplen))
	
	# update running variables
	in_list_n,in_list_sum_prod,in_list_x_s_sum = list_n,list_sum_prod,list_x_s_sum
	in_list_x_sum_s,in_list_y_s_sum,in_list_y_sum_s = list_x_sum_s, list_y_s_sum, list_y_sum_s 

	timepoints = df.index.values

	# compute other rows
	for x in range(1, range_granularity):
		tmp_granularity = steplen*x

		start_time_points.append(timepoints[tmp_granularity])
		n_step_new, result, in_list_n, in_list_sum_prod, in_list_x_s_sum, in_list_x_sum_s, in_list_y_s_sum, in_list_y_sum_s = getFollowingRow(
			T,
			K,
			steplen,
			tmp_granularity,
			in_list_n, in_list_sum_prod, in_list_x_s_sum, in_list_x_sum_s, in_list_y_s_sum, in_list_y_sum_s,
			list_n,list_sum_prod, list_x_s_sum, list_x_sum_s, list_y_s_sum, list_y_sum_s
		)
		row = getValueList(result)
		final.append(row)
	
	return final,start_time_points


def saveContent(results,time_lapse,outputFile,steplen,start_time_points):
	final_json_content = {
		"results":results,
		"time_lapse":time_lapse,
		'start_time_points':start_time_points,
		"params":{"steplen":steplen}
	}

	with open(outputFile, 'w+') as outfile:
		json.dump(final_json_content, outfile)
		print("Results are available at",outputFile)


def main(args):

	random.seed(0)

	inputCsv,outputFile,start,end,edgeThreshold,observedTickers\
	,ifZNorm,ifCompleteGraph,presenceTreshold,\
	classfile,maxEdges,ifClassGraph,overlapThreshold,ifIncludeNegativeEdges,window,winSizeStep,slideStep = parseInputs()

	if winSizeStep <= 1:
		winSizeStep = 5 # default

	time_start = time.time()
	filename = inputCsv.split("/")[-1].split(".")[0]
	df,missing_ticker = loadCsv(inputCsv,observedTickers=observedTickers)
	lenDF = len(df)

	names = df.columns.values
	node_dic = {x:names[x] for x in range(len(names))}

	msts = []
	periods = [] # list of time windows
	metas = []

	if window == -1:
		window_sizes = range(10,lenDF,winSizeStep)
	else:
		window_sizes = [window]

	df,hasClass,class_dic,debug_msg = elaborateData(df,inputCsv,0,0+lenDF-1,observedTickers\
	,ifZNorm,presenceTreshold,ifClassGraph)

	df = df.replace(np.nan,0)
	steplen = int(winSizeStep)

	time_start = time.time()
	final,start_time_points = incremental_cmp(names,df,steplen = steplen)
	time_lapse = time.time()-time_start
	print("---\ntime span for total correlation",round(time_lapse,6))

	saveContent(final,round(time_lapse,6),outputFile,steplen,start_time_points)


if __name__ == '__main__':
  main(sys.argv[1:])




