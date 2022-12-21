import paddle
import numpy as np
import jieba
import matplotlib.pyplot as plt
from tqdm import tqdm

train_data_path="datasets/train.txt"
val_data_path='datasets/dev.txt'
test_data_path='datasets/test.txt'

#读取文件
def openfile(path):
    with open(path,'r',encoding='utf-8') as source:
        lines=source.readlines()
    return lines

train_lines=openfile(train_data_path)
val_lines=openfile(val_data_path)
test_lines=openfile(test_data_path)

#数据预处理
def data_process(datalines, test=False):
    datalist = []
    labellist = []
    for datas in datalines:
        # data,label=datas.strip().split()
        data = datas.strip().split()
        # print(data)
        if test == False:
            labellist.append(data[-1])
            if len(data[:-1]) > 1:
                for i in range(1, len(data[:-1])):
                    data[0] += "," + data[i]
        else:
            if len(data) > 1:
                for i in range(1, len(data)):
                    data[0] += "," + data[i]

        datalist.append(data[0])

    return datalist, labellist

train_data,train_label=data_process(train_lines)
val_data,val_label=data_process(val_lines)
test_data,_=data_process(test_lines,test=True)

#jieba库分词
def jieba_process(datalist):
    data = []
    for datas in tqdm(datalist):
        data.append(jieba.lcut(datas))

    return data

train_data=jieba_process(train_data)
val_data=jieba_process(val_data)
test_data=jieba_process(test_data)

#提取标签并编码
label_set=set()
for label in tqdm(train_label):
    label_set.add(label)

label_dict=dict()
dict_label=dict()
for label in label_set:
    label_dict[label]=len(label_dict)
    dict_label[len(label_dict)-1]=label

#统计标题分布
alllen_dict=dict()
for data in train_data:
    length=len(data)
    if length not in alllen_dict:
        alllen_dict[length]=0
    alllen_dict[length]+=1
alllen_dict = sorted(alllen_dict.items(), key = lambda x:x[0], reverse = False)
print(alllen_dict)
x=[l[0] for l in alllen_dict]
y=[l[1] for l in alllen_dict]
plt.bar(x, y)
plt.xlabel('length')
plt.ylabel('nums')
plt.legend(loc='lower right')
plt.show()

#词库
def build_cropus(data):
    crpous=[]
    for i in range(len(data)):
        crpous.extend(data[i])
    return crpous
allcrpous=build_cropus(train_data+val_data+test_data)
print(len(allcrpous))


# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus, frequency):
    # 首先统计每个不同词的频率（出现的次数），使用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    # 构造3个不同的词典，分别存储，
    word2id_dict = {'<pad>': 0, '<unk>': 1}

    id2word_dict = {0: '<pad>', 1: '<unk>'}

    # 按照频率，从高到低，开始遍历每个单词，并为这个单词构造id
    for word, freq in word_freq_dict:
        if freq > frequency:
            curr_id = len(word2id_dict)
            word2id_dict[word] = curr_id
            id2word_dict[curr_id] = word
        else:
            word2id_dict[word] = 1
    return word2id_dict, id2word_dict, word_freq_dict
word_fre=1
word2id_dict,id2word_dict,word_counts=build_dict(allcrpous,word_fre)
print(len(word2id_dict))
print(len(id2word_dict))
vocab_maxlen=len(word2id_dict)
print('有',len(word2id_dict),'个字被映射到',len(id2word_dict),'个id上') # 字：id

#限制
counts_word_dict = dict()
for word, counts in word_counts:
    if counts not in counts_word_dict:
        counts_word_dict[counts] = 0
    counts_word_dict[counts] += 1

counts_word_dict = sorted(counts_word_dict.items(), key=lambda x: x[0], reverse=False)
# print(counts_word_dict)

x = [l[0] for l in counts_word_dict]
y = [l[1] for l in counts_word_dict]

plt.bar(x[:10], y[:10])
plt.xlabel('frequency')
plt.ylabel('nums')
plt.legend(loc='lower right')
plt.show()

#向量化
tensor_maxlen=15  # 根据统计到的标题长度分布设定
vocab_size=len(id2word_dict)  # 词汇量


def build_tensor(data, dicta, maxlen):
    tensor = []
    for i in range(len(data)):
        subtensor = []
        lista = data[i]
        for j in range(len(lista)):
            index = dicta.get(lista[j])
            subtensor.append(index)

        # 长度限定，不足补0 ；超过则截断
        if len(subtensor) < maxlen:
            subtensor += [0] * (maxlen - len(subtensor))
        else:
            subtensor = subtensor[:maxlen]

        tensor.append(subtensor)
    return tensor
train_tensor=paddle.to_tensor(np.array(build_tensor(train_data,word2id_dict,tensor_maxlen)))
val_tensor=paddle.to_tensor(np.array(build_tensor(val_data,word2id_dict,tensor_maxlen)))
test_tensor=np.array(build_tensor(test_data,word2id_dict,tensor_maxlen))

print(train_label[0])
print(val_label[0])
print(label_dict)
def get_label_tensor(dict,label):
    tensor=[]
    for d in label:
        tensor.append(dict[d])
    return tensor

train_label_tensor=np.array(get_label_tensor(label_dict,train_label))
val_label_tensor=np.array(get_label_tensor(label_dict,val_label))

numclass=len(label_set)
train_label_tensor=paddle.to_tensor(train_label_tensor,dtype='int64')
val_label_tensor=paddle.to_tensor(val_label_tensor,dtype='int64')


#创建数据集
class MyDataset(paddle.io.Dataset):
    def __init__(self, title,lable):
        super(MyDataset, self).__init__()
        self.title = title
        self.lable=lable

    def __getitem__(self, index):
        return self.title[index], self.lable[index]

    def __len__(self):
        return self.title.shape[0]
BATCH_SIZE=128
embed_dim=256
hidden_size=128
train_batch_num=train_tensor.shape[0]//BATCH_SIZE #3482
val_batch_num=val_tensor.shape[0]//BATCH_SIZE #156
print(train_batch_num)
print(val_batch_num)
# 定义数据集
train_dataset = MyDataset(train_tensor,train_label_tensor)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

val_dataset=MyDataset(val_tensor,val_label_tensor)
val_loader=paddle.io.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)


#分类
class Mynet(paddle.nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, data_maxlen, numclass):
        super(Mynet, self).__init__()
        self.numclass = numclass
        self.data_maxlen = data_maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.emb = paddle.nn.Embedding(vocab_size, embed_dim)
        self.gru = paddle.nn.GRU(embed_dim, hidden_size, 2)

        self.l1 = paddle.nn.Linear(hidden_size, 64)
        self.l2 = paddle.nn.Linear(64, 32)
        self.l3 = paddle.nn.Linear(32, self.numclass)

        self.drop = paddle.nn.Dropout(0.5)

    def forward(self, x):
        x = self.emb(x)
        x, states = self.gru(x)
        x = paddle.mean(x, axis=1)

        x = self.drop(x)

        out = paddle.nn.functional.relu(self.l1(x))
        out = self.drop(out)
        out = paddle.nn.functional.relu(self.l2(out))
        out = self.l3(out)
        out = paddle.nn.functional.softmax(out, axis=-1)
        return out

mynet=Mynet(vocab_size,embed_dim,hidden_size,tensor_maxlen,numclass)
paddle.summary(mynet,(128,15),dtypes='int64')
epochs = 20
lr=0.001
log_freq=1000
model_path='./model/train_model'

#训练网络

model=paddle.Model(mynet)

# 为模型训练做准备，设置优化器，损失函数和精度计算方式
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=lr,parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())
model.fit(train_data=train_loader,
          eval_data=val_loader,
          epochs=epochs,
          eval_freq=1,
          save_freq=5,
          save_dir=model_path,

          verbose=1,
          callbacks=[paddle.callbacks.VisualDL('./log')])

#加载模型
infer_model=paddle.Model(Mynet(vocab_size,embed_dim,hidden_size,tensor_maxlen,numclass))
infer_model.load('./model/infer')

with open('result.txt','w',encoding="utf-8") as res:
    for title in test_tensor:
        re = infer_model.predict_batch([[title]])
        #print(re)
        index=paddle.argmax(paddle.to_tensor(re))
        index=int(index[0])
        #print(type(index))
        #print(dict_label[index])
        res.write(dict_label[index]+'\n')
