//迁移学习的思想很简单，在下面有写到。
迁移学习 ： 部分训练好的卷积层不用动(他们的功能在所有类别的图片上都是一样的都是为了提取某一特征)，只动顶层的分类器的dense层和顶层的卷积层（他们的功能让究竟是猫还是狗.....的特征更加明显），然后给予dense层分类。 利用迁移学习可以
减少自己构建深度网络和更新参数的时间。至于为什么看了这程序会了解。其中之一就是不用更新参数。



import tensorflow
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import random
import pathlib

Batch_Size =32

class GET_PATH:
def init(self,path):
self.path = path
self.getdir()
def getdir(self):
self.all_image_path = glob.iglob(self.path)
self.all_image_path = list(self.all_image_path)
random.shuffle(self.all_image_path)
def number(self):
self.all_image_label = []
for var in self.all_image_path:
if pathlib.Path(var).parent.name=="dog":
self.all_image_label.append(0)
else:
self.all_image_label.append(1)
image_count = len(self.all_image_label)

    return self.all_image_path,self.all_image_label,image_count
path ='C:\data\dc_2000\train\\.jpg'
get_path = GET_PATH(path)

all_image_path ,all_image_label,image_count= get_path.number()

def NORMOLIZE_IMAGE(IMAGEPATH):
img = tensorflow.io.read_file(IMAGEPATH)
img_tensor= tensorflow.image.decode_jpeg(img,channels=3)
img_tensor = tensorflow.image.resize_with_crop_or_pad(
img_tensor,120,120)
img_tensor = tensorflow.cast(img_tensor,tensorflow.float32)
img = img_tensor / 255
return img

class Label_and_Normolizeimage:
def init(self,all_image_path,all_image_label):
self.all_image_path = all_image_path
self.all_image_label= all_image_label
def conbine(self):
tensor_path = tensorflow.data.Dataset.from_tensor_slices(self.all_image_path)
image_dataset = tensor_path.map(NORMOLIZE_IMAGE)

    label_dataset=tensorflow.data.Dataset.from_tensor_slices(self.all_image_label)
    dataset = tensorflow.data.Dataset.zip((image_dataset,label_dataset))
    return dataset
obj = Label_and_Normolizeimage(all_image_path,all_image_label)
dataset = obj.conbine()

class DIVIDE_TRAIN:
def init(self,dataset,image_count):
self.dataset = dataset
self.image_count = image_count
def divide(self):
test_count = int(self.image_count*0.2)
train_count = self.image_count - test_count
train_data = self.dataset.skip(test_count)
test_data = self.dataset.take(test_count)
test_data = test_data.batch(batch_size = Batch_Size)
train_data = dataset.shuffle(buffer_size=train_count).batch(batch_size=Batch_Size)
return train_data,test_data,train_count,test_count

divide = DIVIDE_TRAIN(dataset,image_count)
train_data,test_data,train_count,test_cout= divide.divide()

from tensorflow import keras

covn_base = keras.applications.xception.Xception(weights='imagenet',include_top=False)#这个用法一定要会，这个方法完成了迁移学习。要去网上下载xcepion模型，放在user//你的名字//.keras//models文件夹下，没有就新建一个

model = keras.Sequential()
model.add(covn_base)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(256,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

covn_base.trainable = False

model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['acc'], )

history =model.fit(train_data,epochs=10,validation_data=test_data)

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
plt.plot(history.history[metric])
plt.plot(history.history['val_'+metric], '')
plt.xlabel("Epochs")
plt.ylabel(metric)
plt.legend([metric, 'val_'+metric])
plt.show()

这里用的是猫狗数据集2000张那个，用大的数据集也完全可以，图像分类就这几个步骤，
弄懂了逻辑就行了。