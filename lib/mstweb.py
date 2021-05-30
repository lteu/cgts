"""
mst from inout timeseries data

calls example: 
python3 mstweb.py --csv demo_data/ita2001_2016.csv --ifCompleteGraph False
python3 mstweb.py --csv demo_data/ita2010_2020.csv
python3 mstweb.py --csv demo_data/ita2010_2020.csv --ifClassGraph True
python3 mstweb.py --csv demo_data/ita2001_2016_classified.csv --overlapThreshold -0.6
python3 mstweb.py --csv demo_data/ita2001_2016.csv --observedTickers  G.MI,ISP.MI,UCG.MI,FCA.MI,F.MI,ENI.MI,ENEL.MI,TIT.MI,BMED.MI,MB.MI

# @author: toliu [at] unibz.it
"""

import glob
import sys
import os 
import pandas as pd 
import numpy as np  
from mst import *
import time
import networkx as nx
from networkx.readwrite import json_graph
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import os
import json
import collections
import networkx.algorithms.community as nx_comm
from sys import platform # system compatibility switch

def convertDataFrameToClasses(df,class_dic):
	symList = set(df.columns.values.tolist())

	class_symbols = {}

	for tmp_sym in class_dic.keys():
		if tmp_sym not in symList:
			continue

		tmp_class = class_dic[tmp_sym]
		if tmp_class not in class_symbols:
			class_symbols[tmp_class] = [tmp_sym]
		else:
			class_symbols[tmp_class].append(tmp_sym)

	symbs = list(class_symbols.keys())

	for sym in symbs:
		df[sym] = df.loc[:,class_symbols[sym]].mean(axis=1)


	df = df[symbs]

	return df

def analysis(edges,node_dic,class_dic={}):
	G=nx.Graph()
	

	for edge in edges:
		x = edge[0]
		y = edge[1]
		G.add_edge(x,y)

	graph_degrees = G.degree()

	degree_sequence = sorted([d for n, d in G.degree()], reverse=False)  # mac default ?

	degreeCount = collections.Counter(degree_sequence)
	degreeDist = list(degreeCount.items())


	# # remove disconnected nodes to calculate Radius and Graph centers.
	if not nx.is_empty(G):
		largest_cc = max(nx.connected_components(G))
		nodelist = G.nodes()
		to_remove = set(nodelist) - set(largest_cc)
		for nd in to_remove:
			G.remove_node(nd)


	radius,eccentricity,center = "","",""
	if not nx.is_empty(G):
		if nx.is_connected(G):
			radius = nx.radius(G)
			eccentricity = nx.eccentricity(G)
			center = nx.center(G) # The center is the set of nodes with eccentricity equal to radius.


	mod_score = -2

	# check if it makes sense to calculate the modularity score
	aTestNodeName = "" if len(list(node_dic.keys())) == 0 else node_dic[list(node_dic.keys())[0]]
	if bool(class_dic.keys()) and aTestNodeName !="" and aTestNodeName in class_dic:

		inv_dic = {}
		nodelist = G.nodes()
		for n in nodelist:
			converted_num = node_dic[n]
			class_tmp = class_dic[converted_num]
			if class_tmp not in inv_dic:
				inv_dic[class_tmp] = set([n])
			else:
				inv_dic[class_tmp].add(n)
		groups = [x for k,x in inv_dic.items()]
		mod_score = nx_comm.modularity(G, groups)
	return radius,eccentricity,center,degreeDist,mod_score,G


def elaborateData(df,inputCsv,start,end,observedTickers\
	,ifZNorm,presenceTreshold,ifClassGraph):

	debug_msg = ""
	debug_msg += "inputCsv: "+inputCsv+"; "

	hasClass = False
	class_dic = {}
	if "classes" in df.index.values:
		class_dic = df.loc["classes"].to_dict()
		df = df.drop("classes")
		hasClass = True
		debug_msg += "foundClasses: YES; "
	else:
		debug_msg += "foundClasses: NO; "

	df = formatDF(df)

	if start !=  -1 and end != -1 and end !=  0:
		# df  = subFrame(df,start,end)
		df = df.iloc[start:end,:]
		debug_msg += "range: " + str(start)+"-"+str(end)+"; "
	elif start !=  -1: 		
		df = df.iloc[start:,:]
	elif end !=  -1 and end !=  0: 
		df = df.iloc[:-end,:]
	else:
		debug_msg += "range: all; "

	if presenceTreshold:
		# further pruning since some symbols will be disqualify by the number of presence.
		size_pre = len(df.columns.values.tolist())
		df = delInsufficientTickers(df,presenceTreshold)
		size_after = len(df.columns.values.tolist())
		debug_msg += "pruned by percentage of NaNs: from"+str(size_pre)+" to "+str(size_after)+"; "
	else:
		debug_msg += "no pruning of NaNs"


	# z norm
	if ifZNorm != "no":
		# print('yes')
		df  = zNorm(df)
	if hasClass and ifClassGraph:
		df = convertDataFrameToClasses(df,class_dic)
		debug_msg += "classGraph: Yes; "
	else:
		debug_msg += "classGraph: No; "

	return df,hasClass,class_dic,debug_msg

def calculate_class_cover_ratio(edges,node_dic,class_dic):
	num = len(edges) if len(edges) !=0 else 1
	count = 0
	for e in edges:
		n1 = class_dic[node_dic[int(e[0])]]
		n2 = class_dic[node_dic[int(e[1])]]
		if n1 == n2:
			count += 1

	return round(count/num,4)


def parseInputs():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("--csv", help="Input csv of time series")
	parser.add_argument("--start", help="start time point")
	parser.add_argument("--end", help="end time point")
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

	args_ = parser.parse_args()

	inputCsv,start,end,edgeThreshold,observedTickers,ifZNorm,ifCompleteGraph,\
	presenceTreshold,classfile,maxEdges,ifClassGraph,overlapThreshold,ifIncludeNegativeEdges =\
	args_.csv,args_.start,args_.end,args_.edgeThreshold,args_.observedTickers,\
	args_.ifZNorm,args_.ifCompleteGraph,args_.presenceTreshold,args_.classfile,\
	args_.maxEdges,args_.ifClassGraph,args_.overlapThreshold,args_.ifIncludeNegativeEdges

	if None in [inputCsv]:
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

	if None not in [ifCompleteGraph]:
		ifCompleteGraph = True if ifCompleteGraph.lower() == "true" else False
	if None not in [ifClassGraph]:
		ifClassGraph = True if ifClassGraph.lower() == "true" else False
	if None not in [ifIncludeNegativeEdges]:
		ifIncludeNegativeEdges = True if ifIncludeNegativeEdges.lower() == "true" else False

	return inputCsv,int(start),int(end),float(edgeThreshold),\
	observedTickers,ifZNorm,ifCompleteGraph,presenceTreshold,\
	classfile,maxEdges,ifClassGraph,overlapThreshold,ifIncludeNegativeEdges


def delInsufficientTickers(df,presenceTreshold):
	ticker_ids = df.columns.values
	date_ids = df.index.values
	max_nam_allowed = df[ticker_ids[0]].size * (1-presenceTreshold)

	to_del = []
	for ticker in ticker_ids:
		num_nans = df[ticker].isna().sum() 
		if num_nans > max_nam_allowed:
			to_del.append(ticker)

	df=df.drop(columns=to_del)
	return df


def genGML(inputCsv,start,end,edgeThreshold,ifZNorm,ifCompleteGraph,presenceTreshold,maxEdges,ifClassGraph,overlapThreshold,ifIncludeNegativeEdges,outgml):
	filename = inputCsv.split("/")[-1].split(".")[0]
	df,missing_ticker = loadCsv(inputCsv,observedTickers="")

	end = end + 1 # date adjustment
	df,hasClass,class_dic,debug_msg = elaborateData(df,inputCsv,start,end,""\
	,ifZNorm,presenceTreshold,ifClassGraph)


	min_periods = round(len(df) * overlapThreshold) if round(len(df) * overlapThreshold) != 0 else 1

	corr = df.corr(method='pearson',min_periods=min_periods) # df, will be personalized with input

	matrix_unsymmetric = corrToDistance(corr,edgeThreshold=edgeThreshold,ifIncludeNegativeEdges=ifIncludeNegativeEdges) # array like
	if ifCompleteGraph: # full network
		arr = matrix_unsymmetric
		debug_msg += "completeGraph: Yes; "
	else: # MST
		arr = unsymmetricMatrixToMST(matrix_unsymmetric) # df
		debug_msg += "completeGraph: No; "

	names = corr.columns.values
	relevant_one,edges = toJson(corr,arr,names,trim=maxEdges)	

	if not len(relevant_one) == 0:
		radius,eccentricity,center,degreeDist,mod_score,G = analysis(edges,relevant_one,class_dic=class_dic)
	else:
		degreeDist = []
		radius = 0
		center = 0
		eccentricity = []
		mod_score = 0

	rlt = {
		"edges":edges,
		"node_dic":relevant_one,
		"degree_distribution":degreeDist,
		"radius":radius,
		"center":center,
		"eccentricity":eccentricity,
		"mod_score":round(mod_score,4),
		"debug":debug_msg
	}


	if hasClass and not ifClassGraph:
		rlt['symbol_class'] = class_dic
		rlt['classes'] = list(set(class_dic.values()))

		class_cover = calculate_class_cover_ratio(edges,relevant_one,class_dic)

		rlt['class_correctness_ratio'] = class_cover

	return G




def main(args):
	random.seed(0)

	inputCsv,start,end,edgeThreshold,observedTickers\
	,ifZNorm,ifCompleteGraph,presenceTreshold,\
	classfile,maxEdges,ifClassGraph,overlapThreshold,ifIncludeNegativeEdges = parseInputs()


	time_start = time.time()
	filename = inputCsv.split("/")[-1].split(".")[0]

	df,missing_ticker = loadCsv(inputCsv,observedTickers=observedTickers)

	end = end + 1 # date adjustment
	df,hasClass,class_dic,debug_msg = elaborateData(df,inputCsv,start,end,observedTickers\
	,ifZNorm,presenceTreshold,ifClassGraph)

	min_periods = round(len(df) * overlapThreshold) if round(len(df) * overlapThreshold) != 0 else 1

	corr = df.corr(method='pearson',min_periods=min_periods) # df, will be personalized with input

	matrix_unsymmetric = corrToDistance(corr,edgeThreshold=edgeThreshold,ifIncludeNegativeEdges=ifIncludeNegativeEdges) # array like

	if ifCompleteGraph: # full network
		arr = matrix_unsymmetric
		debug_msg += "completeGraph: Yes; "
	else: # MST
		arr = unsymmetricMatrixToMST(matrix_unsymmetric) # df
		debug_msg += "completeGraph: No; "

	names = corr.columns.values

	relevant_one,edges = toJson(corr,arr,names,trim=maxEdges)	

	if not len(relevant_one) == 0:
		radius,eccentricity,center,degreeDist,mod_score,G = analysis(edges,relevant_one,class_dic=class_dic)
	else:
		degreeDist = []
		radius = 0
		center = 0
		eccentricity = []
		mod_score = 0

	rlt = {
		"edges":edges,
		"node_dic":relevant_one,
		"degree_distribution":degreeDist,
		"radius":radius,
		"center":center,
		"eccentricity":eccentricity,
		"mod_score":round(mod_score,4),
		"debug":debug_msg
	}


	if hasClass and not ifClassGraph:
		rlt['symbol_class'] = class_dic
		rlt['classes'] = list(set(class_dic.values()))

		class_cover = calculate_class_cover_ratio(edges,relevant_one,class_dic)

		rlt['class_correctness_ratio'] = class_cover

	print(json.dumps(rlt))



if __name__ == '__main__':
  main(sys.argv[1:])




