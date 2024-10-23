import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd

#损失使用均方损失
loss=nn.MSELoss()

#载入张量形式的数据集
train_features=torch.load('ML_task1/datas/train_features.pt')
test_features=torch.load('ML_task1/datas/test_features.pt')
train_labels=torch.load('ML_task1/datas/train_labels.pt')

in_features=train_features.shape[1]

#构建线性回归的神经网络
#`nn.Sequential`:输入各层的参数，即可搭建一系列的层。使用时，数据会一次通过各个层。
#`nn.Linear`:输入个数、输出个数为参数的线性全连接层
def get_net():
    net=nn.Sequential(nn.Linear(in_features,1))
    return net

#相对误差:防止因数据样本本身的过大或过小而导致误差的绝对值相差较大。
def log_rmse(net,features,labels):
    #`.clamp()`:将输入的张量限制在一个范围内，参数：张量、最小值、最大值
    #`float(inf)`:无穷大
    clipped_preds=torch.clamp(net(features),1,float('inf'))
    #`torch.sqrt()`:计算张量的平方根
    #`torch.log()`:计算张量的元素级自然对数
    rmse=torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()

#训练函数,返回值是模型参数列表，可供后续选择使用
def train(net,train_features,train_labels,test_features,test_labels,num_epoches,learning_rate,weight_decay,batch_size):
    train_ls,test_ls=[],[]
    train_iter=d2l.load_array((train_features,train_labels),batch_size)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)
    #开始训练步骤
    for epoch in range(num_epoches):
        for x,y in train_iter:
            #归零所有被优化的梯度，保证梯度不会累积
            optimizer.zero_grad
            l=loss(net(x),y)
            #计算梯度
            l.backward()
            #根据参数更新模型
            optimizer.step()
            train_ls.append(log_rmse(net,train_features,train_labels))
            if test_labels is not None:
                test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

#模型验证：K折交叉验证
#返回对应第i个k折的训练集和验证集
def get_k_fold_data(k,i,x,y):
    #`assert`:检查某个值是否为真，若为假，则抛出异常
    assert k>1
    fold_size=x.shape[0]//k
    x_train,y_train=None,None
    for j in range(k):
        #`slice`:进行切片操作
        idx=slice(j*fold_size,(j+1)*fold_size)
        x_part,y_part=x[idx,:],y[idx]
        if j==i:
            x_valid,y_valid=x_part,y_part
        elif x_train is None:
            x_train,y_train=x_part,y_part
        else:
            #`torch.cat()`:沿指定维度链接张量
            x_train=torch.cat([x_train,x_part],0)
            y_train=torch.cat([y_train,y_part],0)
    return x_train,y_train,x_valid,y_valid

def k_fold(k,x_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum,valid_l_sum=0,0
    for i in range(k):
        data=get_k_fold_data(k,i,x_train,y_train)
        net =get_net()
        train_ls,valid_ls=train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum+=train_ls[-1]
        valid_l_sum+=valid_ls[-1]
        print('fold %d,train rmse %f,valid rmse %f' % (i,train_ls[-1],valid_ls[-1]))
    return train_l_sum/k,valid_l_sum/k

#调整超参数
def paraAdjust(k,num_epochs,lr,weight_decay,batch_size):
    train_l,valid_l=k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
    print(f'{k}-折验证:平均训练log rmse:{float(train_l):f},'
        f'平均验证log rmse:{float(valid_l):f}')
#调用paraAdjust(k,num_epochs,lr,weight_decay,batch_size)即可评估模型，进行超参数调整

#训练模型进行预测：
def train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size):
    net=get_net()
    train_ls,_=train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    print(f'train log rmse{float(train_ls[-1]):f}')
    #`.detanch()`:分离张量，不再参与梯度计算
    preds=net(test_features).detach().numpy()
    #`pd.Series()`:创建序列对象
    test_data['Saleprice']=pd.Series(preds.reshape(1,-1)[0])
    submission=pd.concat([test_data['Id'],test_data['Saleprice']],axis=1)
    submission.to_csv('ML_task1/results/submission.csv',index=False)

k,num_epochs,lr,weight_decay,batch_size=6,150,9,0,100
Id=input("Id:(int)")
MSSubClass=input("MSSubClass:(int)")
MSZoning=input("MSZoning:(str)")
LotFrontage=input("LotFrontage:(int)")
LotArea=input("LotArea:(int)")
LotConfig=input("LotConfig:(str)")
Neighborhood=input("Neighborhood:(str)")
HouseStyle=input("HouseStyle:(str)")
YearRemodAdd=input("YearRemodAdd:(int)")
Foundation=input("Foundation:(str)")
TotRmsAbvGrd=input("TotRmsAbvGrd:(int)")

test_data={'Id':[Id],
           'MSSubClass':[MSSubClass],
           'MSZoning':[MSZoning],
           'LotFrontage':[LotFrontage],
           'LotArea':[LotArea],
           'LotConfig':[LotConfig],
           'Neighborhood':[Neighborhood],
           'HouseStyle':[HouseStyle],
           'YearRemodAdd':[YearRemodAdd],
           'Foundation':[Foundation],
           'TotRmsAbvGrd':[TotRmsAbvGrd]}

test_data= pd.DataFrame(test_data)
train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)