import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os

#labels -> one_hot
def one_hot(y):
	lb = LabelBinarizer()
	lb.fit(y)
	yy = lb.transform(y)
	return yy

#Z-score标准化
def z_score(x):
	#x = np.log(x)
	x = (x - np.average(x))/np.std(x)
	return x

def adds(path,num):
	files = os.listdir(path)
	nn = len(files)
	Xt = np.zeros((nn, 256, 32))
	for m,n in enumerate(files):
		ms = np.load(path+"/"+n)
		xs = []
		ss = []
		for j in range(32):
			ss = ms[j*256:(j+1)*256]
			xs.append(ss)
		xs = z_score(xs)
		m_s = np.array(xs).transpose(1,0)
		Xt[m,:,:] = m_s

	labels = [num]*nn
	X = Xt.tolist()
	return X,labels
#################################
#训练集合并
def get_train_datas(ks):
	X2,labels_t2 = adds('get_train/datas_mz/AT',0)
	X3,labels_t3 = adds('get_train/datas_mz/PB',1)

	Xtt = [j for j in X2]+[k for k in X3]
	X_tr = np.array(Xtt)

	labels_train = labels_t2 + labels_t3

	y_tr = one_hot(labels_train)

	np.save("train_datas/x_train_"+str(ks)+".npy",X_tr)
	np.save("train_datas/y_train_"+str(ks)+".npy",y_tr)
	print("cross-",ks,"训练数据处理完成")

get_train_datas(1)
