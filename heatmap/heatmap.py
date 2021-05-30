"""

Baseline approach to computing a heatmap.
Functions offered by pandas are used.

python3 heatmap.py --csv ../data/Ita-2019-2020-July_classified_small.csv  --out out/rlt.json
# python3 heatmap.py --csv ../data/Ita-2019-2020-July_classified_small.csv  --presenceTreshold 0.8 --overlapThreshold 0.8  --ifIncludeNegativeEdges false  --winSizeStep 3 --edgeThreshold 0.1 --out out/tmp.json


Mandatory parameters:
--inputCsv
--outputFile

"""

import glob
import sys
import os 
import pandas as pd 
import numpy as np 
import time
import networkx as nx
from networkx.readwrite import json_graph
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import os
import json
import collections

root_arr = os.path.realpath(__file__).split('/')[:-2]
root = '/'.join(root_arr) 
sys.path.append(root+"/lib")

from mst import corrToDistance,zNorm,subFrame,dataFrameFromFile,unsymmetricMatrixToMST,toJson,loadCsv,formatDF
from mstweb import convertDataFrameToClasses,analysis,elaborateData,calculate_class_cover_ratio,delInsufficientTickers

def parseInputs():
	"""
	Parse command line parameters
	"""
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("--csv", help="Input csv of time series")
	parser.add_argument("--out", help="Output json")
	parser.add_argument("--start", help="start time point")
	parser.add_argument("--end", help="end time point")
	parser.add_argument("--window", help="window size")
	parser.add_argument("--edgeThreshold", help="Edge threshold")
	parser.add_argument("--observedTickers", help="Tickers to observe")
	parser.add_argument("--ifZNorm", help="Tickers to observe")
	# 13.05.2020
	parser.add_argument("--ifCompleteGraph", help="1 for complete 0 otherwise")
	parser.add_argument("--presenceTreshold", help="Percentage of available time points above which a symbol is qualified")
	parser.add_argument("--overlapThreshold", help="percentage of overlap for Pearson correlation")
	parser.add_argument("--classfile", help="A csv which defines the class of each ticker")
	parser.add_argument("--maxEdges", help="Max number of edges to show")
	parser.add_argument("--ifClassGraph", help="A graph of classes or symbols")

	# 13.06.2020
	parser.add_argument("--ifIncludeNegativeEdges", help="PLZ indicate true or false")


	# 20.10.2020
	parser.add_argument("--winSizeStep", help="Step of window size default 1")

	parser.add_argument("--slideStep", help="Step size of a sliding window,  default 1")

	args_ = parser.parse_args()

	inputCsv,outputFile,start,end,edgeThreshold,observedTickers,ifZNorm,ifCompleteGraph,\
	presenceTreshold,classfile,maxEdges,ifClassGraph,overlapThreshold,ifIncludeNegativeEdges,window,winSizeStep,slideStep =\
	args_.csv,args_.out,args_.start,args_.end,args_.edgeThreshold,args_.observedTickers,\
	args_.ifZNorm,args_.ifCompleteGraph,args_.presenceTreshold,args_.classfile,\
	args_.maxEdges,args_.ifClassGraph,args_.overlapThreshold,args_.ifIncludeNegativeEdges,args_.window,args_.winSizeStep,args_.slideStep

	if None in [inputCsv]:
		print (__doc__)
		sys.exit(0)


	if None in [outputFile]:
		print (__doc__)
		sys.exit(0)


	if not winSizeStep:
		winSizeStep = 1
	else:
		winSizeStep = int(winSizeStep)

	if not start:
		start = -1
	if not end:
		end = -1
	if not window:
		window = -1 # default 100 time points
	if not edgeThreshold:
		edgeThreshold = 0.0
	if not observedTickers:
		observedTickers = ""
	else:
		observedTickers = observedTickers.replace("[space]"," ").replace("[quote]","'")

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
		maxEdges = 1500

	if not slideStep:
		slideStep = 1

	if None not in [ifCompleteGraph]:
		ifCompleteGraph = True if ifCompleteGraph.lower() == "true" else False
	if None not in [ifClassGraph]:
		ifClassGraph = True if ifClassGraph.lower() == "true" else False
	if None not in [ifIncludeNegativeEdges]:
		ifIncludeNegativeEdges = True if ifIncludeNegativeEdges.lower() == "true" else False

	return inputCsv,outputFile,int(start),int(end),float(edgeThreshold),\
	observedTickers,ifZNorm,ifCompleteGraph,presenceTreshold,\
	classfile,maxEdges,ifClassGraph,overlapThreshold,ifIncludeNegativeEdges,window,winSizeStep,slideStep



def process_df(msts,subdf,class_dic,overlapThreshold,edgeThreshold,ifIncludeNegativeEdges,ifCompleteGraph,maxEdges,ifClassGraph):
	"""
	Compute all graph information given a MST generated from a specific time range
	"""
	min_periods = round(len(subdf) * overlapThreshold) if round(len(subdf) * overlapThreshold) != 0 else 1
	tmp_corr = subdf.corr(method='pearson',min_periods=min_periods) # will be personalized with input
	matrix_unsymmetric = corrToDistance(tmp_corr,edgeThreshold=edgeThreshold,ifIncludeNegativeEdges=ifIncludeNegativeEdges)

	if ifCompleteGraph: # full network
		arr = matrix_unsymmetric
	else: # MST
		arr = unsymmetricMatrixToMST(matrix_unsymmetric) # array of selected edges

	names = tmp_corr.columns.values
	relevant_one,edges = toJson(tmp_corr, arr, names,trim=maxEdges)	 # further filtering and get node names

	radius,eccentricity,center,degreeDist,mod_score,graph = analysis(edges,relevant_one,class_dic=class_dic) # full analysis

	msts.append(edges)
	

	rlt = {"node_dic":relevant_one,
	"degree_distribution":degreeDist,
	'first_degree_score':firstDegreeScore(degreeDist),
	"radius":radius,
	"center":center,
	"mod_score":round(mod_score,4),
	"eccentricity":eccentricity}
	if len(class_dic.keys())>0 and not ifClassGraph:
		rlt['symbol_class'] = class_dic
		rlt['classes'] = list(set(class_dic.values()))

	# local_periods =[]
	local_periods = [subdf.iloc[[0]].index.values.tolist()[0],subdf.iloc[[-1]].index.values.tolist()[0]] # related time ranges
	return msts,rlt,local_periods


def firstDegreeScore(degreeDist):
	'''
	First degree formula
	'''
	if len(degreeDist) == 0:
		return np.nan
	firstDegreeScore = degreeDist[0][1] / sum([x[1] for x in degreeDist])
	return firstDegreeScore

def main(args):
	random.seed(0)

	inputCsv,outputFile,start,end,edgeThreshold,observedTickers\
	,ifZNorm,ifCompleteGraph,presenceTreshold,\
	classfile,maxEdges,ifClassGraph,overlapThreshold,ifIncludeNegativeEdges,window,winSizeStep,slideStep = parseInputs()

	time_start = time.time()

	filename = inputCsv.split("/")[-1].split(".")[0]
	df,missing_ticker = loadCsv(inputCsv,observedTickers=observedTickers)

	lenDF = len(df)
	names = df.columns.values
	node_dic = {x:names[x] for x in range(len(names))}
	msts = []
	tmp_corr = []
	periods = []
	metas = []

	if window == -1:
		window_lengths = list(range(10,lenDF,winSizeStep)) # 10 is the minimum window length
	else:
		window_lengths = [window]


	# start_points = list(range(0,lenDF-window_len,1))


	for window_len in window_lengths:
		print('Computing window length',window_len,'(Top:',window_lengths[-1],')')

		row_meta = []
		row_periods = []
		for start_point in range(0,lenDF-window_len,5):
			
			subdf = subFrame(df,start_point,start_point+window_len) # interested period
			
			if len(subdf) == 0:
				continue


			# preprocessing
			subdf,hasClass,class_dic,debug_msg = elaborateData(df,inputCsv,start_point,start_point+window_len-1,observedTickers\
			,ifZNorm,presenceTreshold,ifClassGraph)


			# mst and features
			msts,rlt,tmp_periods = process_df(msts,subdf,class_dic,overlapThreshold,edgeThreshold,ifIncludeNegativeEdges,ifCompleteGraph,maxEdges,ifClassGraph)
			row_meta.append(rlt)
			row_periods.append(tmp_periods)
			print(f'\r{start_point}', end=' ', flush=True)

		metas.append(row_meta)
		periods.append(row_periods)

	results = {
		# "msts":msts,
		"metas":metas,
		"node_dic":node_dic,
		"debug_msg":debug_msg,
		"periods":periods
	}

	params = {
		"inputCsv":inputCsv,
		"start":start,
		"end":end,
		"edgeThreshold":edgeThreshold,
		"observedTickers":observedTickers,
		"ifZNorm":ifZNorm,
		"ifCompleteGraph":ifCompleteGraph,
		"presenceTreshold":presenceTreshold,
		"classfile":classfile,
		"maxEdges":maxEdges,
		"ifClassGraph":ifClassGraph,
		"overlapThreshold":overlapThreshold,
		"ifIncludeNegativeEdges":ifIncludeNegativeEdges,
		"window_lengths":window_lengths
	}
	# print(json.dumps(rlt))

	time_lapse = time.time()-time_start
	print("---\ntime span for total correlation",round(time_lapse,6))

	final_json_content = {
		"results":results,
		"time_lapse":time_lapse,
		"params":params
	}

	with open(outputFile, 'w+') as outfile:
		json.dump(final_json_content, outfile)




if __name__ == '__main__':
  main(sys.argv[1:])




