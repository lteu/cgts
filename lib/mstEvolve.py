"""
# @author: toliu [at] unibz.it
"""

import glob
import sys
import os 
import pandas as pd 
import numpy as np  
from mst import addToColorGraph,corrToDistance,zNorm,subFrame,dataFrameFromFile,unsymmetricMatrixToMST,toJson,loadCsv,formatDF,arrToNX
from mstweb import elaborateData,analysis
import time
import networkx as nx
from networkx.readwrite import json_graph
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import os
import json
import collections



def parseInputs():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("--csv", help="Input csv of time series")
	parser.add_argument("--start", help="start time point")
	parser.add_argument("--end", help="end time point")
	parser.add_argument("--edgeThreshold", help="Edge threshold")
	parser.add_argument("--observedTickers", help="Tickers to observe")
	parser.add_argument("--ifZNorm", help="Tickers to observe")

	parser.add_argument("--step", help="time step")
	parser.add_argument("--window", help="window size")

	# 13.05.2020
	parser.add_argument("--ifCompleteGraph", help="1 for complete 0 otherwise")
	parser.add_argument("--presenceTreshold", help="Percentage of available time points above which a symbol is qualified")
	parser.add_argument("--overlapThreshold", help="overlap value for Pearson correlation")
	
	parser.add_argument("--classfile", help="A csv which defines the class of each ticker")
	parser.add_argument("--maxEdges", help="Max number of edges to show")
	parser.add_argument("--ifClassGraph", help="A graph of classes or symbols: true or false")

	# 13.06.2020
	parser.add_argument("--ifIncludeNegativeEdges", help="PLZ indicate true or false")

	args_ = parser.parse_args()

	inputCsv,start,end,edgeThreshold,observedTickers,\
	ifZNorm,ifCompleteGraph,presenceTreshold,classfile,\
	maxEdges,ifClassGraph,step,window,overlapThreshold,ifIncludeNegativeEdges  =\
	args_.csv,args_.start,args_.end,args_.edgeThreshold,args_.observedTickers,\
	args_.ifZNorm,args_.ifCompleteGraph,args_.presenceTreshold,args_.classfile,\
	args_.maxEdges,args_.ifClassGraph,args_.step,args_.window,args_.overlapThreshold,\
	args_.ifIncludeNegativeEdges


	if None in [inputCsv,step,window]:
		print (__doc__)
		sys.exit(0)

	if not start:
		start = -1
	if not end:
		end = -1
	if not edgeThreshold:
		edgeThreshold = 0.0
	if not observedTickers:
		observedTickers = ""

	if presenceTreshold:
		presenceTreshold = float(presenceTreshold)

	if overlapThreshold:
		overlapThreshold = float(overlapThreshold)
		if  overlapThreshold <= 0.0 :
			overlapThreshold = 0.0
		elif overlapThreshold >= 1.0:
			overlapThreshold = 1.0
	else:
		overlapThreshold = 0.0

	if maxEdges:
		maxEdges = int(maxEdges)
	else:
		maxEdges = 2000

	if None not in [ifCompleteGraph]:
		ifCompleteGraph = True if ifCompleteGraph.lower() == "true" else False
	if None not in [ifClassGraph]:
		ifClassGraph = True if ifClassGraph.lower() == "true" else False
	if None not in [ifIncludeNegativeEdges]:
		ifIncludeNegativeEdges = True if ifIncludeNegativeEdges.lower() == "true" else False

	return inputCsv,int(start),int(end),float(edgeThreshold),\
	observedTickers,ifZNorm,ifCompleteGraph,presenceTreshold,\
	classfile,maxEdges,ifClassGraph,int(step),int(window),overlapThreshold,ifIncludeNegativeEdges


def main(args):
	random.seed(0)


	inputCsv,start,end,edgeThreshold,observedTickers\
	,ifZNorm,ifCompleteGraph,presenceTreshold,\
	classfile,maxEdges,ifClassGraph,step,window,overlapThreshold,ifIncludeNegativeEdges  = parseInputs()

	time_start = time.time()
	filename = inputCsv.split("/")[-1].split(".")[0]
	df,missing_ticker = loadCsv(inputCsv,observedTickers=observedTickers)

	df,hasClass,class_dic,debug_msg = elaborateData(df,inputCsv,start,end,observedTickers\
	,ifZNorm,presenceTreshold,ifClassGraph)


	lenDF = len(df)


	names = df.columns.values
	node_dic = {x:names[x] for x in range(len(names))}
	# print(df)
	# sys.exit()
	msts = []
	tmp_corr = []
	periods = []
	metas = []
	c = 0
	for x in range(0,lenDF-window,step):
		subdf = subFrame(df,x,x+window)

		if len(subdf) == 0:
			continue

		min_periods = round(len(subdf) * overlapThreshold) if round(len(subdf) * overlapThreshold) != 0 else 1
		# min_periods = 30
		tmp_corr = subdf.corr(method='pearson',min_periods=min_periods) # will be personalized with input
		matrix_unsymmetric = corrToDistance(tmp_corr,edgeThreshold=edgeThreshold,ifIncludeNegativeEdges=ifIncludeNegativeEdges)


		if ifCompleteGraph: # full network
			arr = matrix_unsymmetric
		else: # MST
			arr = unsymmetricMatrixToMST(matrix_unsymmetric) # df
		
		# print(class_dic)
		# sys.exit()
		names = tmp_corr.columns.values
		relevant_one,edges = toJson(tmp_corr, arr,names,trim=maxEdges)	
		# print(arr,names)
		# sys.exit()

		if ifClassGraph:
			class_dic = {}
		radius,eccentricity,center,degreeDist,mod_score = analysis(edges,relevant_one,class_dic=class_dic)

		msts.append(edges)
		

		rlt = {"node_dic":relevant_one,
		"degree_distribution":degreeDist,
		"radius":radius,
		"center":center,
		"mod_score":round(mod_score,4),
		"eccentricity":eccentricity}
		if hasClass and not ifClassGraph:
			rlt['symbol_class'] = class_dic
			rlt['classes'] = list(set(class_dic.values()))

		metas.append(rlt)
		c += 1

		periods.append([subdf.iloc[[0]].index.values.tolist()[0],subdf.iloc[[-1]].index.values.tolist()[0]])

	
	rlt = {
		"msts":msts,
		"metas":metas,
		"node_dic":node_dic,
		"debug_msg":debug_msg,
		"periods":periods
	}


	print(json.dumps(rlt))

if __name__ == '__main__':
  main(sys.argv[1:])
# print('HI from python')