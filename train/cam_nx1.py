from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras.utils import np_utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
from keras.layers import Dense, Activation,Dropout,Convolution1D,MaxPooling1D,Flatten
import numpy as np
t = 2

def fun(ys):
	yy = []
	for i in ys:
		yy = yy+i.tolist()
	return yy 

#(n,256,32)
x_train = np.load("train_datas/x_train_"+str((int(t)-1))+".npy") #(n,向量长度，通道数) 
y_train = fun(np.load("train_datas/y_train_"+str((int(t)-1))+".npy")) #one_hot编码向量 (n,)
x_test = np.load("train_datas/X_test_"+str((int(t)-1))+".npy")
y_test = fun(np.load("train_datas/y_test_"+str((int(t)-1))+".npy"))

y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)

#################modeling#######################
# 建立序贯模型
model = Sequential()                                           #256*32

model.add(Convolution1D(                                       #256*64
    filters=64,
    kernel_size=2,
    padding='same',
    strides=1,
    input_shape=(256,32))) 

model.add(MaxPooling1D(                                       #128*64
    pool_size=2,
    strides=2,
    padding='same')) 

model.add(Convolution1D(                                       #128*128
    filters=128,
    kernel_size=2,
    padding='same',
    strides=1)) 

model.add(MaxPooling1D(                                       #64*128
    pool_size=2,
    strides=2,
    padding='same')) 

model.add(Convolution1D(                                      #64*256
    filters=256,
    kernel_size=2,
    padding='same',
    strides=1)) 

model.add(MaxPooling1D(                                       #32*256
    pool_size=2,
    strides=2,
    padding='same')) 

model.add(Convolution1D(                                      #32*512
    filters=512,
    kernel_size=2,
    padding='same',
    strides=1)) 

model.add(MaxPooling1D(                                       #16*512
    pool_size=2,
    strides=2,
    padding='same')) 

model.add(Convolution1D(                                      #16*1024
    filters=1024,
    kernel_size=2,
    padding='same',
    strides=1)) 

model.add(MaxPooling1D(                                       #8*1024
    pool_size=2,
    strides=2,
    padding='same')) 

# Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())                                           #8192

model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#包含100个神经元的全连接层，激活函数为ReLu，dropout比例为0.5
model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 包含10个神经元的输出层，激活函数为Softmax
model.add(Dense(units=2))
model.add(Activation('softmax'))

# 输出模型的参数信息
model.summary()

#######################cconfiguration############
# 配置模型的学习过程
adam = optimizers.Adam(lr=0.00001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

#######################training###################
model.fit(x_train,y_train,batch_size=32,epochs=100)

#######################evaluate###################
score=model.evaluate(x_test,y_test)
print('Test accuracy:', score[1])
