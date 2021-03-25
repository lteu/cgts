# core functions related to MST
# @author: toliu [at] unibz.it

import glob
import sys
import os 
import pandas as pd 
import numpy as np  
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import time
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import zscore


def checkDelimFromLine(line):
	delim = ","
	if "\t" in line:
		delim = "\t"
	elif "," in line:
		delim = ","
	elif ";" in line:
		delim = ";"
	elif " " in line:
		delim = " "
	return delim

def loadCsv(testFile,observedTickers=""):
	delim = ","
	with open(testFile) as ff:

		for line in ff:
			if "\t" in line:
				delim = "\t"
			elif "," in line:
				delim = ","
			elif ";" in line:
				delim = ";"
			elif " " in line:
				delim = " "
			break

	df = pd.read_csv(testFile, index_col = 0, header=0, delimiter=delim, dtype='unicode')

	missing_tickers = ""
	if observedTickers != "" :
		observedTickersArr = set([x for x in observedTickers.split(",") if x.strip() != ""])
		original_nodenames = df.columns.values.tolist()
		missing_tickers = set(observedTickersArr) - set(original_nodenames)
		selected_tickers = [x for x in original_nodenames if x not in missing_tickers and x in observedTickersArr]
		df = df[selected_tickers]

	return df, ",".join(list(missing_tickers))

def formatDF(df):
	
	df = df.replace('\\N',np.NaN)
	df = df.replace('NaN',np.NaN)
	df = df.astype(float)
	# how to remove non-numeric values ?

	

	return df

def dataFrameFromFile(testFile,observedTickers=""):
	delim = ","
	with open(testFile) as ff:

		for line in ff:
			if "\t" in line:
				delim = "\t"
			elif "," in line:
				delim = ","
			elif ";" in line:
				delim = ";"
			elif " " in line:
				delim = " "
			break

	df = pd.read_csv(testFile, index_col = 0, header=0, delimiter=delim)

	# delete class
	if "classes" in df.index.values:
		classes = df.loc["classes"]
		df = df.drop("classes")


	observedTickersArr = set([x for x in observedTickers.split(",") if x.strip() != ""])
	missing_tickers = ""
	if observedTickers != "" :
		original_nodenames = df.columns.values.tolist()
		missing_tickers = set(observedTickersArr) - set(original_nodenames)
		selected_tickers = [x for x in original_nodenames if x not in missing_tickers and x in observedTickersArr]
		df = df[selected_tickers]

	df = df.replace('\\N',np.NaN)
	df = df.replace('NaN',np.NaN)

	df = df.astype(float)
	# how to remove non-numeric values ?

	return df, ",".join(list(missing_tickers))

def zNorm(df):
	for col in df.columns.values:
		df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

	return df

def converVal(x, edgeThreshold):
	val = 2-x
	if val < edgeThreshold:
		val = 0.0
	return val


def corrToDistance(corr_input,edgeThreshold=0.0,ifIncludeNegativeEdges=True):

	corr = corr_input.copy()
	
	# distance formula
	corr = corr.apply(lambda x:2-abs(x)) if ifIncludeNegativeEdges else corr.apply(lambda x:2-x)

	# make unqualified pairs unreachable
	corr[corr > 2 - edgeThreshold] = 0
	corr = corr.fillna(0)
	
	# buttom part to 0
	matrix_unsymmetric = corr.values.tolist()
	for idx,row in enumerate(matrix_unsymmetric):
		for x in range(0,idx):
			matrix_unsymmetric[idx][x] = 0.0

	# diagonal to 0
	for i in range(len(matrix_unsymmetric)):
		matrix_unsymmetric[i][i] = 0.0

	return matrix_unsymmetric


def arrToNX(arr):
	G=nx.Graph()
	edges = []
	for x in range(len(arr)):
		G.add_node(x)
		for y in range(len(arr[x])):
			if float(arr[x][y]) != 0.0:
				edges.append([x,y,round(2-arr[x][y],2)])

	
	for edge in edges:
		x = edge[0]
		y = edge[1]
		G.add_edge(x,y)
	return G

def toJson2(corr):
	print('in')
	return 1,2
def toJson(corr, arr, names , trim=None ):
	orig_corr = corr.values.tolist()
	edges = []

	for x in range(len(arr)):
		for y in range(len(arr[x])):
			if float(arr[x][y]) != 0.0: # 0.0 are unreachable pair points
				edges.append([x,y,round(orig_corr[x][y],2)]) 


	edges.sort(key=lambda x: x[2],reverse=True)

	if trim:
		edges = edges[:trim]

	relevant_nodes = []
	for e in edges:
		relevant_nodes.append(e[0])
		relevant_nodes.append(e[1])

	if len(names) != 0:
		relevant_one = {x:names[x] for x in relevant_nodes}
	else:
		relevant_one = {x:x for x in relevant_nodes}



	return relevant_one,edges


def subFrame(df,start,end):
	reduceDF = df.iloc[start:end,:]
	return reduceDF

def unsymmetricMatrixToMST(matrix_unsymmetric):
	Tcsr = minimum_spanning_tree(matrix_unsymmetric)
	arr= Tcsr.toarray().astype(float)
	return arr


def main(args):
	

	time_start = time.time()

	testFile = args[0]

	start = -1
	end = -1
	if len(args) > 1:
		start,end = int(args[1]),int(args[2])


	filename = testFile.split("/")[-1].split(".")[0]

	df = dataFrameFromFile(testFile)


	if start !=  -1 and end != -1:
		df  = subFrame(df,start,end)

	# z norm
	df  = zNorm(df)

	# corr
	corr = df.corr(method='pearson',min_periods=30) # will be personalized with input

	matrix_unsymmetric = corrToDistance(corr)

	# tree
	arr = unsymmetricMatrixToMST(matrix_unsymmetric)
	
	
if __name__ == '__main__':
  main(sys.argv[1:])





