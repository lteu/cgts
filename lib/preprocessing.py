"""
preprocess a file to working file

--conversion: none, weekly, monthly
--manipulation: none, gain, loggain

@author: toliu [at] unibz.it
"""

import glob
import sys
import os 
import pandas as pd 
import numpy as np  
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import os
import json
import math

from io import StringIO
from mst import corrToDistance,\
zNorm,subFrame,dataFrameFromFile,\
unsymmetricMatrixToMST,toJson,checkDelimFromLine


import datetime
from datetime import  timedelta


def parseInputs():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("--file", help="Input file")
	parser.add_argument("--classFile", help="Class file")
	parser.add_argument("--conversion", help="conversion option")
	parser.add_argument("--manipulation", help="manipulation option")
	parser.add_argument("--output", help="output direction")
	args_ = parser.parse_args()

	file,conversion,manipulation,output,classFile = args_.file,args_.conversion,args_.manipulation,args_.output,args_.classFile

	if None in [file,output]:
		print (__doc__)
		sys.exit(0)

	if not manipulation:
		manipulation = "none"
	if not conversion:
		conversion  = "none"

	return file,conversion,manipulation,output,classFile


def calculate_loggain(a,b):
	if np.isnan(a) or np.isnan(b):
		return np.nan
	elif a > 0 and b > 0: # pay attention to this
		return round(math.log(b) - math.log(a),5)
	else:
		return np.nan

def calculate_gain(a,b):
	if np.isnan(a) or np.isnan(b) or a == 0:
		return np.nan
	else:
		return round((b - a)/a,5)

def checkFormat(lines):
	fileformat = "none"
	first_line = lines[0]
	delim = checkDelimFromLine(first_line)
	parts = first_line.split(delim)
	if "Adj Close" in  lines[0] or len(parts) < 5:
		fileformat = "long"
	elif "-" in  lines[2] or "/" in lines[2]:
		fileformat = "csv"
	elif len(lines[0].split(",")) == 2:
		fileformat = "classes"

	return fileformat


def manipulate(df,manipulation):
	df_ref = df.copy()
	# print(manipulation)
	ticker_ids = df.columns.values
	date_ids = df.index.values
	for ticker in ticker_ids:
		for i in range(1,len(date_ids)):
			this_date = date_ids[i]
			prev_date = date_ids[i-1]
			a = df_ref[ticker][prev_date]
			b = df_ref[ticker][this_date]
			if manipulation =="loggain":
				val = calculate_loggain(a,b)
			elif manipulation =="gain":
				val = calculate_gain(a,b)
			else:
				pass

			df[ticker][this_date] = val

	df = df.drop([date_ids[0]]) # in case of gain, the first row is not necessary

	return df


def longToCsv(datafile):

	# check delim
	with open(datafile) as f:
		first_line = f.readline()

	delim = checkDelimFromLine(first_line)

	# remove empty lines
	with open(datafile) as file:
		lines_raw = file.readlines()
	lines = [x for x in lines_raw if x.replace(delim,"").strip() != ""]
	str_content = StringIO("".join(lines))


	df = pd.read_csv(str_content, index_col = 0, header=0, delimiter=delim)

	columnames = df.columns.values
	sym_name = ""
	value_name = ""

	for name in columnames:
		tmp_elm = str(df[name].iloc[1])
		if not tmp_elm.isdigit():
				sym_name = name

	if "Adj Close" in columnames:
		value_name = 'Adj Close'
	elif len(columnames) == 2:
		reduced_colname = [x for x in columnames if x != sym_name]
		value_name = reduced_colname[0]
	
	# wrong format , empty DF
	if sym_name == "" or value_name == "":
		return pd.DataFrame()

	sym_ids = df[sym_name].unique()
	date_ids = df.index.unique()
	date_ids = list(date_ids)
	date_ids.sort()
	sym_dictionary = {x:{} for x in sym_ids}
	for date,row in df.iterrows():
		sym_dictionary[row[sym_name]][date] = row[value_name]

	sym_ids.sort()
	result = []
	for sym in sym_ids:
		info = sym_dictionary[sym]
		row = [sym]
		for index,date in enumerate(date_ids):
			if date in info:
				row.append(info[date])
			else:
				row.append('NaN')
		result.append(row)

	df = pd.DataFrame(result,columns=(['']+date_ids))
	df.set_index([''], inplace = True)  
	df = df.transpose()
	df = df.replace('NaN',np.NaN)


	return df

def longToCsv_old(file):
	df = pd.read_csv(file, header=0, encoding = 'unicode_escape',delimiter=',') 
	sym_ids = df['Ticker'].unique()
	date_ids = df['Date'].unique()
	date_ids = list(date_ids)
	date_ids.sort()
	ticker_dictionary = {x:{} for x in sym_ids}
	count = 0
	for i,row in df.iterrows():
		date = row['Date']
		ticker_dictionary[row['Ticker']][date] = row['Adj Close']
	sym_ids.sort()
	result = []
	for ticker in sym_ids:
		info = ticker_dictionary[ticker]
		row = [ticker]
		for index,date in enumerate(date_ids):
			if date in info:
				row.append(info[date])
			else:
				row.append('NaN')

		result.append(row)

	df = pd.DataFrame(result,columns=(['']+date_ids))
	df.set_index([''], inplace = True)  
	df = df.transpose()
	df = df.replace('NaN',np.NaN)

	return df



def getValidDaysForWeekly(date_obj, time_obj_fistday):
	rlt = []
	for x in range(0,3):
		tmp_time_obj = date_obj-datetime.timedelta(x)
		# print(tmp_time_obj,time_obj_fistday)
		if tmp_time_obj.date() >= time_obj_fistday.date():
			rlt.append(tmp_time_obj.strftime("%Y-%m-%d"))

	return rlt


def dailyToWeekly(df):
	df = df.replace('\\N',np.NaN)
	df = df.replace('NaN',np.NaN)
	df = df.astype(float)
	df.index = pd.to_datetime(df.index)
	df = df.resample('W-THU').last()
	return df

def dailyToMonthly(df):
	df = df.replace('\\N',np.NaN)
	df = df.replace('NaN',np.NaN)
	df = df.astype(float)
	df.index = pd.to_datetime(df.index)
	df = df.resample('M').last()
	return df

def weeklyAvg(df):
	df = df.replace('\\N',np.NaN)
	df = df.replace('NaN',np.NaN)
	df = df.astype(float)
	df.index = pd.to_datetime(df.index)
	df = df.resample('W-MON').mean()
	return df

def monthlyAvg(df):	
	df = df.replace('\\N',np.NaN)
	df = df.replace('NaN',np.NaN)
	df = df.astype(float)
	df.index = pd.to_datetime(df.index)
	df = df.resample('M').mean()
	return df

def convert(df,conversion):
	ticker_ids = df.columns.values
	date_ids = df.index.values

	if conversion == "weekly":
		df = dailyToWeekly(df)
		# print(df)
	elif conversion == "monthly":
		df = dailyToMonthly(df)
		# print(df)
	elif conversion == "weeklyAvg":
		df = weeklyAvg(df)
	elif conversion == "monthlyAvg":
		df = monthlyAvg(df)
	else:
		pass

	return df


def addClassToDF(df,df_class):
	t1 = df.columns.values.tolist()
	t2 = df_class.index.values.tolist()

	labelName = df_class.columns.values.tolist()[0]
	to_del = set(t1)-set(t2)

	# delete filtered out	
	df = df.drop(columns=list(to_del))
	remained_tickers =  df.columns.values.tolist()

	# class labelss
	classes = {}
	for t in remained_tickers:
		val = df_class.loc[t][labelName]
		classes[t] = val

	new_row = pd.DataFrame(classes,index =['classes']) 
	df = pd.concat([new_row, df])
	return df

# ---- utility functions -----

def next_weekday(d, weekday):
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0: # Target day already happened this week
        days_ahead += 7
    return d + datetime.timedelta(days_ahead)

def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)



def generateWorkingFile(file,conversion,manipulation,output,classFile):
	with open(file) as myfile:
		lines = [next(myfile) for x in range(3)]
	fileformat = checkFormat(lines)

	if fileformat == "long":		
		df = longToCsv(file)
	elif fileformat == "csv":
		df,missing_tickers = dataFrameFromFile(file)
	else:
		df = None

	if not df.empty and conversion != "none":
		df =  convert(df,conversion)

	if not df.empty and manipulation != "none":
		df = manipulate(df,manipulation)

	if classFile:
		df_class = pd.read_csv(classFile, index_col = 0, header=0, delimiter=',')
		df = addClassToDF(df,df_class)


	# check nan percentage after conversion
	percent_missing = df.isnull().sum() * 100 / len(df)
	percent_missing = percent_missing.mean()
	msg,outpath = "ok",output

	if not df.empty:
		if percent_missing > 70:
			msg = "The input file does not support the current preprocessing request; it yields too many NaNs."
		else: 
			if os.path.isfile(outpath):
				os.remove(outpath)
			df.to_csv(outpath,sep=',')
	else:
		msg = "Format error"

	message ={"outpath":outpath,"conversion":conversion,"manipulation":manipulation,"msg":msg}
	print(json.dumps(message))
	
def main(args):

	file,conversion,manipulation,output,classFile = parseInputs()

	with open(file) as myfile:
		lines = [next(myfile) for x in range(3)]
	fileformat = checkFormat(lines)

	if fileformat == "long":		
		df = longToCsv(file)
	elif fileformat == "csv":
		df,missing_tickers = dataFrameFromFile(file)
	else:
		df = None

	if not df.empty and conversion != "none":
		df =  convert(df,conversion)

	if not df.empty and manipulation != "none":
		df = manipulate(df,manipulation)

	if classFile:
		df_class = pd.read_csv(classFile, index_col = 0, header=0, delimiter=',')
		df = addClassToDF(df,df_class)


	# check nan percentage after conversion
	percent_missing = df.isnull().sum() * 100 / len(df)
	percent_missing = percent_missing.mean()

	msg,outpath = "ok",output

	if not df.empty:
		if percent_missing > 70:
			msg = "The input file does not support the current preprocessing request; it yields too many NaNs."
		else: 
			if os.path.isfile(outpath):
				os.remove(outpath)
			df.to_csv(outpath,sep=',')
	else:
		msg = "Format error"

	message ={"outpath":outpath,"conversion":conversion,"manipulation":manipulation,"msg":msg}
	print(json.dumps(message))

if __name__ == '__main__':
  main(sys.argv[1:])




