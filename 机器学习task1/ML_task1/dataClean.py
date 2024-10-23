import numpy as np
import pandas as pd
import os
import torch

#读取train.csv和test.csv文件
train_data=pd.read_csv("ML_task1/datas/train.csv")
test_data=pd.read_csv("ML_task1/datas/test.csv")

#抽取所需特征值(即所有行以及1、2、3、4、10、12、16、20、29、55列)，并将train集和test集合并存于features之中
#`.iloc[]`:选择数据的方法，第一个参数为所需行，第二个参数为所需列。
#`pd.concat()`:沿特定轴（默认行，即纵向）合并pandas对象。
features=pd.concat((train_data.iloc[:,[1,2,3,4,10,12,16,20,29,55]],test_data.iloc[:,[1,2,3,4,10,12,16,20,29,55]]))

#识别数值型的，将其归一化，即均值为0，方差为一的样本
#1. 识别数值型
#`.index`即对象的索引。该条将数值型的索引存在numtype中
numtype=features.dtypes[features.dtypes !='object'].index
#2. 将数值型的归一化
#`.apply()`:可指定其执行自定义函数
#`lambda`:创建匿名函数的关键字，简单到无需用def
#`.mean()`:求平均数
#`.std()`:求标准差
features[numtype]=features[numtype].apply(
    lambda x:(x-x.mean())/(x.std())
)
#3. 将not a number转换为0
#`.fillna`:填充NaN值
features[numtype]=features[numtype].fillna(0)

#离散值用独热编码
#`.get_dummies()`:将分类变量转换为虚拟/指示变量
#`dummy_na`:若为True，则为缺失值创建一个特别类
features=pd.get_dummies(features,dummy_na=True)

#将数据从pandas格式提取Numpy格式，转换为张量表示。并重新分给train和test集，将标签集单独提取出来。
n_train=train_data.shape[0]
#`.values`:将数据转换为Numpy ndarray对象
train_features=torch.tensor(features[:n_train].values,dtype=torch.float32)
test_features=torch.tensor(features[n_train:].values,dtype=torch.float32)
train_labels=torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)
#将张量保存到本地
torch.save(train_features,'ML_task1/datas/train_features.pt')
torch.save(test_features,'ML_task1/datas/test_features.pt')
torch.save(train_labels,'ML_task1/datas/train_labels.pt')