
# coding: utf-8

# In[5]:

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.regularizers import l2


# In[7]:

# 载入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# (60000,28,28)
print('x_shape:',x_train.shape)
# (60000)
print('y_shape:',y_train.shape)
# (60000,28,28)->(60000,784)
x_train = x_train.reshape(x_train.shape[0],-1)/255.0
x_test = x_test.reshape(x_test.shape[0],-1)/255.0
# 换one hot格式
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# 创建模型
model = Sequential([
        Dense(units=500,input_dim=784,bias_initializer='one',activation='tanh',kernel_regularizer=l2(0.0003)),
        Dropout(0.5),
        Dense(units=300,bias_initializer='one',activation='tanh',kernel_regularizer=l2(0.0003)),
        Dropout(0.5),
        Dense(units=10,bias_initializer='one',activation='softmax',kernel_regularizer=l2(0.0003))
    ])

# 定义优化器
adam = Adam(lr=0.001) 

# 定义优化器，loss function，训练过程中计算准确率
model.compile(
    optimizer = adam,
    loss = 'categorical_crossentropy',
    metrics=['accuracy'],
)

# 训练模型
model.fit(x_train,y_train,batch_size=32,epochs=10)

# 评估模型
loss,accuracy = model.evaluate(x_test,y_test)

print('\ntest loss',loss)
print('accuracy',accuracy)


# In[ ]:



