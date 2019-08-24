import numpy as np
import os

paths = ['get_train/datas/AT','get_train/datas/PB']

def get_rand_vecs():
	lis = np.load("get_train/datas/vecs.npy")
	n = 4096
	indexs = np.random.choice(lis.shape[0],n,replace=False)
	lists = lis[indexs]
	np.save("get_train/save_npy/vecs_key.npy",lists)
	return lists

def get_vecs(path,lists,low,high,ks):
	files = os.listdir(path)
	for l in files:
		res = []
		arr = []
		f = np.load(path+'/'+l)
		x1 = f[:,0]
		x2 = f[:,1]
		me1,me2 = np.percentile(x1, [low, high])
		for i in range(len(x1)):
			if x1[i] > me1 and x1[i] < me2:
				arr.append(i)
		ff = f[arr]
		for j in lists:
			ss = ff[np.argwhere(ff[:,1]==j)][:,0]
			if ss.tolist() == []:
				ss = [0]
			res.append(np.mean(ss))
			res.append(np.std(ss))
		np.save("get_train/datas_mz/"+path[-2:]+"/"+str(ks)+l,res)
		print("get_train/datas_mz/"+path[-2:]+"/"+str(ks)+l)

def get_vec_mz(ks,key,l,h):
	for path in paths:
		get_vecs(path,key,l,h,ks)
	print("cross-",ks,"特征向量提取完成")

l = 1
h = 99

for i in range(1):
	key = get_rand_vecs()
	get_vec_mz(i,key,l,h)
