# main函数你们看会少_ _,别忘了加
import numpy
import tensorflow
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import re

# 文本观点分类的步骤
# 1处理text 2 编号观点 3 text向量化 4 构成 x，y
# 这一个函数用来编号观点
def Number_label(text):
if text == 'positive':
text = '1'
text = int(text)
elif text == 'negative':
text = '0'
text = int(text)
else:
text = '2'
text = int(text)
return text

token = re.compile('[A-Za-z]+ | [!,.()?]')

//这一个函数用来处理data.text 例如 某一条text处理前[ i am zhangsan] 处理后---->['i','am','zhangsan']
def Norm_text(text):
new_text = token.findall(text)
new_text = [word.replace(" ","").lower() for word in new_text]
return new_text

//这一个函数将每一行text简单向量化 然后 Embdding编码进一步向量化
def Vector(text):
new_text = []
for word in text:
word = str(data_set[word])
new_text .append(int(word))
return new_text

class Handle_Data:
def init(self,data_path):
self.data = pd.read_csv(data_path)
self.handle()
def handle(self):
global data_set
global max_len
global word_sum
data = self.data[['airline_sentiment','text']]
data_positive = data[data.airline_sentiment == 'positive']
data_negative = data[data.airline_sentiment == 'negative']
data_neutral = data[data.airline_sentiment == 'neutral']
#均匀化 都取一样的数目
data_positive = data_positive
data_negative = data_negative.iloc[0:len(data_positive)]
data_neutral = data_neutral.iloc[0:len(data_positive)]

    data = pd.concat([data_positive,data_negative,data_neutral])
    data = data.sample(len(data))
    data['reviews'] = data.airline_sentiment.apply(Number_label)
    del  data['airline_sentiment']
    data['text'] = data.text.apply(Norm_text)
    all_word = []
    for text in data.text:
        for word in text:
            all_word.append(word)
    data_word = list(set(all_word))
    word_sum  = len(data_word) + 1  #总的单词数 + 0 这个字符 为用到的所有的单词数目，将0看成一个特殊单词，因为要用他填充向量
    data_set = dict((word, data_word.index(word) + 1) for word in data_word)
    data['text'] = data.text.apply(Vector)
    vector_list = []
    for vector in data.text:
        vector_list.append(len(vector))
    max_len = max(vector_list)
    data_end = keras.preprocessing.sequence.pad_sequences(data.text.values, maxlen=max_len)
    ouputs = data.reviews.values

    return  data_end,ouputs,word_sum,max_len  #input [[one],  #output [one_label,two_label]
                                   # [two],]  #输入的数集， 输出的标签 ， 用到的所有的单词数目，  data.text中所有text比较，得到的最大的text长度。用最长的为基准来填充0，让每一行text都变为一样长度
class Model:
def init(self,data_end,outputs,word_sum,max_len):
self.data_end = data_end
self.outputs = outputs
self.word_sum = word_sum
self.max_len = max_len
def model(self):
model = tensorflow.keras.Sequential()
model.add(layers.Embedding(input_dim=self.word_sum, output_dim=50, input_length=self.max_len))#将文本向量化，每一条text都将变成向量注意参数
model.add(layers.LSTM(64))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(self.data_end, self.outputs, epochs=30, batch_size=100, validation_split=0.2)
return history

#这一个类用来描述损失函数和准确率
class plot_graphs:
def init(self,history, metric):
self.history = history
self.metric = metric
def Plt(self):
plt.plot(self.history.history[self.metric],color = 'red')
plt.plot(self.history.history['val_'+self.metric], '')
plt.xlabel("Epochs")
plt.ylabel(self.metric)
plt.legend([self.metric, 'val_'+self.metric])
plt.show()

if name == 'main':

handle_data = Handle_Data("c://data/Tweets.csv")
data_end,ouputs,word_sum,max_len  = handle_data.handle()
model = Model(data_end,ouputs,word_sum,max_len )
history = model.model()