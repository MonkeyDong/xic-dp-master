import os
import re
import numpy as np

'''
import xlrd
workbook1 = xlrd.open_workbook('table.xlsx')
booksheet1 = workbook1.sheet_by_index(0)
samples = []
rows = booksheet1.nrows
for i in range(1,rows):
	ff = booksheet1.cell_value(i,0)
	samples.append(ff)
'''

all = os.listdir('IPX0001046000part')

def split_(ii):
	for i,j in enumerate(ii[::-1]):
		if j not in ['A','T','P','B']:
			continue
		else:
			if j in ['A','T']:
				return 1
			if j == 'P' and ii[::-1][i+1] == 'R':
				continue		
			else:
				return 0

pb_all = []
at_all = []
for i in all:
	if split_(i) == 1:
		at_all.append(i)
	if split_(i) == 0:
		pb_all.append(i)	


def get_re(ff):
	for i,j in enumerate(ff[::-1]):
		if j not in ['A','T','P','B']:
			continue
		if j == 'P' and ff[::-1][i+1] == 'R':
			continue		
		else:
			return ff[:-i]	
	 
dic = {}
for i in pb_all:
	ind = get_re(i)
	if ind not in dic.keys():
		dic[ind] = [i]
	else:
		dic[ind].append(i)

redic = {}	
for i in at_all:
	ind = get_re(i)
	if ind not in redic.keys():
		redic[ind] = [i]
	else:
		redic[ind].append(i)

def load_data(file_path):
	ll = np.array([9,12])
	bars = []
	f = open(file_path,'r')
	data = f.readlines()
	for ii in data[1:-1]:
		cc = ii.split('\n')[0]
		cc = cc.split('\t')
		cc = (np.array(list(map(float,cc))))[ll]
		#print(cc)
		cc[1] = cc[1]+0.0001
		cc[1] = float(re.findall(r"\d{1,}?\.\d{2}", str(cc[1]))[0])
		bars.append(cc)
	bars = sorted(bars, key=lambda arr: arr[1])
	return bars

for i in dic.items():
	n = 0
	for j in i[1]:
		print(j)
		gg = load_data('IPX0001046000part/'+j)
		if n == 0:
			rr = gg
			n = 1
		else:			
			rr = np.concatenate((rr,gg),axis = 0)
	np.save('AT/'+str(i[0])+'.npy',rr)

for i in redic.items():
	n = 0
	for j in i[1]:
		gg = load_data('IPX0001046000part/'+j)
		if n == 0:
			rr = gg
			n = 1
		else:
			rr = np.concatenate((rr,gg),axis = 0)
	np.save('PB/'+str(i[0])+'.npy',rr)







