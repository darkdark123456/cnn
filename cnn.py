main函数少_ _别忘了加
步骤 ： 1 处理标签 2 图片矩阵归一化 3构造tf.data作为输入 3 构建模型 4输入模型 5 优化参数
import numpy
import tensorflow
import matplotlib.pyplot as plt
import pathlib
import cv2
import random
import glob

Batch_size = 32

这一个类用来处理目标路径的图片
class Get_path:
def init(self,path):
self.path = path
self.getdir()
self.number()

def getdir(self):#这一个函数是为了得到图像的目录  例如c://data//2_class//airplane//*.jpeg
    self.all_image_path = glob.iglob(self.path)
    self.all_image_path = list(self.all_image_path)
    random.shuffle(self.all_image_path)
    return self.all_image_path

def number(self):#这一个函数是为了处理标签，让0代表飞机，1代表卫星  顺便返回图片的数量
    self.all_image_label = []
    for var in self.all_image_path:
        if pathlib.Path(var).parent.name == "airplane": # 当前路径的父节点的名字
            self.all_image_label.append(0)
        else:
            self.all_image_label.append(1)
    self.imag_cout = len(self.all_image_label)
    return self.all_image_label,self.imag_cout #返回所有的图片的类别例如[1,0,0]代表[卫星，飞机，飞机]
这一个函数用来归一化图片
def load_preprosess_image(imagpath):
img_raw = tensorflow.io.read_file(imagpath)
img_tensor = tensorflow.image.decode_jpeg(img_raw,channels=3)
img_tensor = tensorflow.image.resize_with_crop_or_pad(img_tensor,256,256)#给解码后的图片修改尺寸
img_tensor = tensorflow.cast(img_tensor, tensorflow.float32)
img =img_tensor / 255
return img

这个类来处理归一化后的图片和编号
class Label_correspond_normolizeimage:
def init(self,all_image_path,all_image_label):
self.all_image_path = all_image_path
self.all_label = all_image_label
self.conbine()
def conbine(self):
tensor_path = tensorflow.data.Dataset.from_tensor_slices(self.all_image_path)# 所有的路径先转换成tf.data类型
imag_dataset = tensor_path.map(load_preprosess_image)# 1 遍历tensor——path的每一个元素 2放入load_preprosess_image处理 3 将处理后的图片放在一个列表中
#imag_dataset存放的是所有归一化后的图片
label_dataset = tensorflow.data.Dataset.from_tensor_slices(self.all_label)#将标签对应的编号变为tf.data类型
# #归一化前 图片标签与编号一一对应 归一化后图片与标签仍然一一对应。因为图片在列表里的位置始终没动，只是归一化
dataset=tensorflow.data.Dataset.zip((imag_dataset,label_dataset))#让归一化后图片和编号一一对应
return dataset

这个类用来划分数据集和测试集
class Divide_train_test:
def init(self,dataset,image_count):
self.dataset = dataset
self.image_count = image_count
self.divide()
def divide(self):
test_count = int(self.image_count *0.2)
train_count = self.image_count -test_count
train_data = self.dataset.skip(test_count)
test_data = self.dataset.take(test_count)
test_data = test_data.batch(batch_size=Batch_size) # 每次处理test——data为batch——size个
train_data = dataset.shuffle(buffer_size=train_count).batch(batch_size=Batch_size).repeat(6) # 全部打乱，每次取batch_size训练 重复六次
return train_data,test_data,train_count,test_count

class CNN_Model:
def init(self,train_data,test_data,train_count,test_count):
self.train_data = train_data
self.test_data = test_data
self.train_count = train_count
self.test_count = test_count
self.Model()

def Model(self):
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'))
    model.add(tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tensorflow.keras.layers.MaxPooling2D())

    model.add(tensorflow.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tensorflow.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tensorflow.keras.layers.MaxPooling2D())

    model.add(tensorflow.keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(tensorflow.keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(tensorflow.keras.layers.MaxPooling2D())

    model.add(tensorflow.keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(tensorflow.keras.layers.GlobalAveragePooling2D())

    model.add(tensorflow.keras.layers.Dense(1024, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc'],

    )

    steps_per_epoch = self.train_count // Batch_size,
    validation_steps = self.test_count // Batch_size,


    history = model.fit(self.train_data,epochs=5,validation_data=self.test_data,validation_steps=validation_steps[0],steps_per_epoch=steps_per_epoch[0])
    return  0
if name == 'main':
newget = Get_path('C:\data\2_class\\.jpg')#表示任何名字在这里，放.不行，不知道为什么
all_image_path = newget.getdir()
all_image_label,imag_count = newget.number()
label_correspond_normolizeimage = Label_correspond_normolizeimage(all_image_path,all_image_label)
dataset = label_correspond_normolizeimage.conbine()
# <ZipDataset shapes: ((256, 256, 3), ()), types: (tf.float32, tf.int32)>
# 图片矩阵 类别 float32型 离散
divide = Divide_train_test(dataset,imag_count)
train_data,test_data,train_count,test_count = divide.divide()
#tf.Tensor: id = 870, shape = (32#一共32张, 256, 256, 3), dtype = float32, numpy =[one,two,...] , [one_label,two_label,..]

cnnmodel = CNN_Model(train_data,test_data,train_count,test_count)
下面的代码用来可视化，需要自行添加在最后一个类里
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()