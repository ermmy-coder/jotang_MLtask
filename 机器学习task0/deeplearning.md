焦糖工作室学习笔记
========
刘伊贝 2023091204024
-------

>简单说明学习情况：在寒假跟着实验室学习过，第一次接触了机器学习，但为期两周只是自学了一些概念，十月学习了《动手学深度学习》第三章，并跟着配套网课走了一下，其中有些已经知道的概念就没有整理了。


# 《动手学深度学习》第三章 深度学习基础

## 3.1线性回归
### 3.1.1基本要素
* **模型**（model）
    - 建立输入`x1`和`x2`的表达式：`y=x1w1+x2w2+b`
    - 权重(weight):`w1`、`w2`
    - 偏差(bias):`b`
* **模型训练**(model training)：通过数据寻找参数
* **训练数据**（training data set）
    - 样本(sample)
    - 标签(lable)
    - 特征(feature)
* **损失函数**(loss function)
    - 常用平方函数（square loss）
    - `l(w1,w2,b)=(1/n)Σli(w1,w2,b)`
    `li(w1,w2,b)=(1/2)(w1xi+w2x2+b-yi)^2`
* **优化算法**
    - 常用*小批量随机梯度下降*
    `wt=wt-1-η（l对wt-1的偏导）`
        - `η`：学习率，即走多长
        - `偏导`:走的方向
    - 小批量：随机采样`b`个样本来近似损失
    - 梯度下降：沿反梯度方向更新求解参数
    - 调参：调节超参数（hyperparameter）
        - 批量大小（bach size）
        - 学习率(learning rate)
* 模型预测
### 3.1.2表示方法
1. 神经网络图
2. 矢量计算表达式：提高计算效率

### 3.1.3线性回归的从零实现
>此部分为跟着书上的代码跟着**手搓**了一遍
>收获如下：
>* 线性回归的损失是凸函数，所以最优解满足偏导数为0，故线性的最优解为显示解。（但要用机器学习解决的一般没有显示解）
>* 每一个特征值都是对结果影响的影响因素，`w`为其加权值。即线性回归是对n维输入的加权，外加偏差。
>* 线性回归可以看作是单层神经网络。

``` 
#%matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
import torch

#生成数据集
num_inputs=2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2
features=nd.random.normal(scale=1,shape=(num_examples,num_inputs))
lables=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
lables+=nd.random.normal(scale=0.01,shape=lables.shape) #服从正态分布的噪声（即无意义的干扰）

print(features[0])
print(lables[0])
#可视表示
def use_svg_display():
    #用矢量图表示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    #设置图的尺寸
    plt.rcParams['figure.figsize']=figsize

set_figsize()
plt.scatter(features[:,1].asnumpy(),lables.asnumpy(),1);

#读取数据集
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)#样本读取顺序是随机的
    for i in range(0,num_examples,batch_size):
        j=nd.array(indices[i:min(i+batch_size,num_examples)])
        yield features.take(j),lables.take(j)

batch_size=10

for X,Y in data_iter(batch_size,features,lables):
    print(X,Y)
    break

#定义初始化模型参数
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

#定义模型
def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X,w)+b
#定义损失函数
def square_loss(y_hat,y):
    """均方损失"""
    return (y_hat-y.reshape(y_hat.shape))**2/2
#定义优化算法
def sgd(params,lr,batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

#训练过程
lr=0.03
num_epochs=3
net=linreg
loss=square_loss

for epoch in range(num_epochs):
    for X,Y in data_iter(batch_size,features,lables):
        l=loss(net(X,w,b),Y)#`X`和`Y`的小批量损失
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l=loss(net(features,w,b),lables)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')
```

## 3.2softmax回归 
### 3.2.1基础知识
1. 分类和回归的**区别**：
- 均方损失：某一类的置信度为1，其他为0（**有多类**）
- 无校验比例：正确类的**置信度**远大于非类（大于某一阈值）
- 校验比例：输出匹配**概率**
    - `yihat=exp(oi)/Σexp(ok)`即softmax运算（softmax运算即将`oi`转化为`yihant`）
    - 做指数的好处是保证非负
    - 概率`y`和`yhat`的区别作为损失
    - `W`是`特征值数*样本个数`的权重参数矩阵，`b`是`1*样本数`的偏差参数矩阵，`oi=W*xi+b`
3. *交叉熵损失*
    - **目的**：衡量两个概率（预测和标号）的区别
    - **实现**：
        1. 构造第i个元素为1，其余为0的真实向量y
        2. `H(yi,yihat)=-Σyi*log(yihat)`
        3. 交叉熵损失函数定义：`l(δ)=ΣH(yi,yihat)/n`
    - **实质**：对真实类的预测概率的负对数（因为非类乘的是真实类中的0）
    - 梯度就是真实类和预测类的差别：故可令左右相等
    `l(y,yhat)对oi求偏导=softmax(o)i-yi`
    - 不用均方：只需要预测类别正确即可，平方差太过严格

## 损失函数
1. 均方损失（L2 loss）
- `l(y.y')=[(y-y')**2]/2`
- 缺点：梯度变化不稳定
2. 绝对值损失（L1 loss）
- `l（y,y'）=|y-y'|`
- 缺点：零点处不可导，在优化末期劣势大
3. 鲁伯函数（Huber’s Robust loss）
- 在接近原点处为均方损失，在距离较远处为绝对值损失。

## 3.3感知机
### 3.3.1基础知识
1.  与上两种回归的**区别**：二分类
2. 输入`x`、权重`w`、偏差`b`;输出`o=δ(<w,x>+b)`，`δ(x)=1(if x>0),=0(otherwise)`
3. **损失函数**：`l(y,x,w)=max（0,-y<w,x>）`
4. **训练过程原理**等价于使用批量大小为1的梯度下降，若输出`o`大于零，则分类正确，否则加`oixi`给`w`，加`oi`给`y`
5. **收敛定理**：
- 数据在半径`r`内
- `o(w·x+b)>=ρ`即出现一个余量——分界面
- 感知机保证在`（r^2+1）/ρ^2`步后收敛
### 3.3.2多层感知机【解决XOR问题】
>*XOR问题*即无法用线性分界面进行分类的问题
1. **基本原理**：利用简单函数进行两种标准的分类，再用一个简单函数将两种标准联系起来
2. **单隐藏层**（Hidden layer）,隐藏层的大小是超参数
3. **激活函数**：
- 一定不是线性的
- `h=W1·x+b1`
`o=w2T·h+b2`
- 几种常见的激活函数
    - *Sigmoid激活函数*【简单经典】
        - 目标：将输入投影至(0,1)；曲线光滑
        - `sigmoid(x)=1/[1+exp(-x)]`
        - 解释：可保证其处于（0，1）之间
    - *Tanh激活函数*
        - 目标：将输入投影至（-1，1）；曲线光滑
        - `tanh(x)=[1-exp(-2x)]/[1+exp(-2x)]`
    - *ReLU激活函数(rectified linear unit)
        - `ReLU(x)=max(x,0)`
        - 好处：计算很快且能实现非线性
4. **多类分类**：在softmax()回归中间加了一层隐藏层，即对`o`再做softmax()
5. **多隐藏层**：顾名思义
>超参数：隐藏层数、每层隐藏层大小
>调整经验：逐层减少 or 先扩后缩




        




