房价预测实验报告
=======
2023091204024 刘伊贝
-------

# 一、实验过程
## 1.数据集
### 1.1数据集分析
#### 1.1.1特征值 （80个）

>加粗的特征值为分析后可作为有效特征值的。

1. **MSSubClass(住宅类型)**
- 16种输入
- 输入为`20-190`的整数，输入分布不均匀 
2. **MSZoning(住宅分区)**
- 8种输入
- 输入为字母型的标签
3. **LotFrontage（与街道的直线英尺距离）**
- `21-313`的整数、`NAN`（not a number,默认为0）
4. **LotArea（住宅面积）（单位为平方英尺）**
- `1300-215245`的整数
5. Street（是否铺路/砾石路）
- 99.59%均为`pave`，不具区分性，舍去。
6. Alley（小巷是否铺路/砾石路）
- 93.7%均为`NA`（没有小巷），不具区分性，舍去。
7. LotShape（住宅形状）
- 76.53%均为`REG`，且远超其他类，不具区分性，舍去。
8. LandContour（平整度）
- 89.79%均为`lvl`，不具区分性，舍去。
9. Utilities（可使用的基础设施）
- 99.93%均为`AllPub`，不具区分性，舍去。
10. **LotConfig（土地配置）**
- 5种输入
- 输入为字母型的标签
11. LandSlope（住宅倾斜度）
- 96.66%均为`Gtl`，不具区分性，舍去。
12. **Neighborhood（在市区范围内的物理位置）**
- 25种输入
- 输入为字母型的标签
13. Condition1
- 86.78%均为`Normal`，且远超其他类，不具区分性，舍去。
14. Condition2
- 98.97%均为`Normal`，且远超其他类，不具区分性，舍去。
15. BldgType（住宅类型）
- 83.24%均为`1Fam`，且远超其他类，不具区分性，舍去。
16. **HouseStyle（住宅风格）**
- 8种输入
- 输入为字母型的标签
17. OverallQual（对装修和材料的评估）
- `1-10`之间的整数，集中于`4-8`之间
- 有些细枝末节，暂且舍去。
18. OverallCond（对住宅整体评估）
- `1-10`之间的整数
- 集中在`5-7`之间，数据不够分散，不具区分性，舍去。
19. YearBuilt（修建年份）
- `1872-2010`的整数
- 由于有改建日期，以改建日期为准，故舍去。
20. **YearRemodAdd（改建日期）**
- `1950-2010`的整数
21. RoofStyle（屋顶风格）
- 78.15%均为`Gamble`，且客观逻辑来讲对房价影响属于细枝末节，暂且舍去。
22. RoofMatl
- 98.22%均为`CompShg`，不具区分性，舍去。
29. **Foundation（基础类型）**
- 6种输入
- 输入为字母型的标签
39. Heating（加热方式）
- 97.81%均为`GasA`，不具区分性，舍去。
46. GrLivArea（地面居住面积）
- `334-5642`之间的整数
- 已考虑房屋总面积，故暂且舍去此因素。
55. **TotRmsAbvGrd（总的房间数）**
- `2-14`之间的整数，主要集中在`5-8`之间。
75. MiscFeature（其他未覆盖的杂项功能）
- 杂项不予考虑。
76. MiscVal（杂项的值）
- 杂项不予考虑。
77. MoSold（出售月）
- 无关项。
78. YrSold（出售年）
- 无关项。
79. SaleType（出售方式）
- 无关项。
80. SaleCondition（出售状态）
- 无关项。

>23. Exterior1st（房屋外墙覆盖物）
>24. Exterior2nd
>25. MasVnrType（砌体贴面型）
>26. MasVnrArea（砌体贴面面积）（以平方英尺计）
>27. ExterQual（外部材料质量）
>28. ExterCond（外部材料形状）
>30. BsmtQual（地下室高度）
>31. BsmtCond（地下室的一般情况）
>32. BsmtExposure（花园暴露情况）
>33. BsmtFinType1（地下室完工等级）
>34. BsmtFinSF1（1型完工面积）
>35. BsmtFinType2（地下室完工面积等级）
>36. BsmtFinSF2（2型完工面积）
>37. BsmtUnfSF（未完工的地下室面积）
>38. TotalBsmtSF（地下室总面积）
>40. HeatingQC（加热质量和条件）
>41. CentralAir（中央空调）
>42. Electrical（供电系统）
>43. 1stFlrSF（一楼大小）
>44. 2ndFlrSF（二楼大小）
>45. LowQualFinSF（低质量成品面积）
>47. BsmtFullBath（地下全浴室）
>48. BsmtHalfBath（地下半浴室）
>49. FullBath(地上全浴室)
>50. HalfBath（地上半浴室）
>51. Bedroom（地上卧室）
>52. Bedroom（地上卧室数）
>53. KitchenAbvGr（地上厨房数）
>54. KitchenQual（厨房质量）
>56. Functional（家庭功能？）
>57. Fireplaces（壁炉数量）
>58. FireplaceQu（壁炉质量）
>59. GarageType（车库类型）
>60. GarageYrBlt（车库修建年份）
>61. GarageFinish（车库的完成度）
>62. GarageCars（汽车容量大小）
>63. GarageArea（车库面积）
>64. GarageQual（车库质量）
>65. GarageCond（车库条件）
>66. PavedDrive（道路车道）
>67. WoodDeckSF（木制甲板面积）
>68. OpenPorchSF（开放式门廊面积）
>69. EnclosedPorch（封闭式门廊面积）
>70. 3SsnPorch（三季门廊面积）
>71. ScreenPorch（屏风们面积）
>72. PoolArea（泳池面积）
>73. PoolQC（泳池质量）
>74. Fence（栅栏质量）
>以上特征以客观逻辑来讲对房价影响均属于细枝末节，暂且舍去。（若后续优化模型再做补充）

#### 1.1.2输出项
**SalePrice**
- `34900-755000`之间的整数

#### 1.1.3总结
1. 10个特征值
2. 输入分析：
- 1和55为特定的几种整数型的输入。
- 3、4和20为某区间之间的任意整数型。
- 2、10、12、16、29为特定几种字符型的输入。

### 1.2数据清洗和归一化
#### 1.2.1思路
1. 删除ID值、无需特征值。（或者抽取所需特征值）
2. 若为数值类的输入，则归一化为均值为0，方差为一。
3. 若为离散的分类文本，则用独热编码替换.
4. 将数据从pandas格式转换为张量表示。
#### 1.2.2所遇问题
1. **问题**：读取csv相对地址时，`.py`文件放在了`ML_task1`文件夹里面，与`.py`文件同级的`datas`文件夹内的`train.csv`文件的相对地址写为了`"datas/train.csv"`,结果找不到该文件。
**解决过程**：使用`os.getcwd()`发现当下目录为`ML_task1`文件夹所在的`diveIntoDeepLearning`文件夹。因而正确的相对地址应该为`"ML_task1/datas/train.csv"`。

## 2.模型
### 2.1模型构建
1. 一些选择：
- 损失：均方损失（理由：简单易于表示）
- 模型：使用线性回归模型（理由：输出为一个值）
- 使用相对误差而非绝对误差
- 使用Adam优化器（相当于更光滑的sgd，但对学习率的敏感度更低，调参难度更小）
- 验证时使用
2. 训练过程：
循环每一个`epoch`：归零梯度——>将样本数据经过一次神经网络得到输出`yhat`——>将预测值`yhat`与样本中的标签值`y`求损失（均方损失）——>计算梯度——>根据梯度更新优化模型——>将模型记录下来
### 2.2模型评估
#### 2.2.1 选择
- k折交叉验证：即将样本数据集分为`k`份，每次取`k-1`份作为训练集，1份作为验证集。如此进行`k`次训练，每份都可以做一次验证集。

#### 2.2.2 评估及调参过程
**首批参数**：k,num_epochs,lr,weight_decay,batch_size=5,100,5,0,64
**结果**：
fold 0,train rmse 0.636631,valid rmse 0.626347
fold 1,train rmse 0.663544,valid rmse 0.660488
fold 2,train rmse 0.679405,valid rmse 0.700948
fold 3,train rmse 0.653584,valid rmse 0.629597
fold 4,train rmse 0.657158,valid rmse 0.674643
5-折验证：平均训练log rmse:0.658064,平均验证log rmse:**0.658405**

1. 调整学习率ir：
- **ir=3**:平均训练log rmse:1.148807,平均验证log rmse:1.148975 
>尝试调小ir，发现log rmse更大，舍弃此操作。
- **ir=7**:平均训练log rmse:0.319146,平均验证log rmse:0.319252 
>尝试调大ir，发现log rmse有明显变小，操作有效。
- **ir=9**:平均训练log rmse:0.295304,平均验证log rmse:**0.295394**  
>逐步调整，发现ir在8.95-9.05之间最大，故取**ir=9**。

2. 调整批次大小batch_size:
- batch_size=30:平均训练log rmse:0.565374,平均验证log rmse:0.568230 
>调小batch_size，发现更大，舍弃。
- batch_size=80:平均训练log rmse:0.369957,平均验证log rmse:0.370223 
>较为大幅调大batch_size，发现稍微变大了，也许操作可行。
>缩小幅度后经过多次调整，发现对结果影响不大，最优超参数不变，即依旧取**batch_size=64**。

3. 调整k:
- k=3:
3-折验证：平均训练log rmse:0.336958,平均验证log rmse:0.338471
>调小k,log rmse变大，说明k对结果有影响，继续调整。
- k=6:
6-折验证：平均训练log rmse:0.295449,平均验证log rmse:**0.295202** 
>反复调整发现k=6时结果稍有减小，多次验证发现有稳定的下降，故取**k=6**。

4. 调整训练次数num_epochs:
- num_epochs=200:平均训练log rmse:0.557725,平均验证log rmse:0.562551 
>调大epochs，发现结果更差。
- num_epochs=150,batch_size=100:
平均训练log rmse:0.295325,平均验证log rmse: **0.295100**
>同时调节epoch和batch参数，发现有一定提升，故取**num_epochs=150,batch_size=100**

**最终参数**：k,num_epochs,lr,weight_decay,batch_size=6,150,9,0,100

# 二、实验成果

## 实验成果
本模型利用`Kaggle波士顿房价数据集`，训练了一个波士顿地区的房价预测模型，最终模型的均方误差控制在`0.29-0.33`之间。

## 操作说明
下载`ML_task1`，激活`gluon_env.yaml`虚拟环境，先后运行`dataClean.py`和`model.py`。

## 输入
```
Id:(int)1463
MSSubClass:(int)45
MSZoning:(str)RL
LotFrontage:(int)72
LotArea:(int)10933
LotConfig:(str)Corner
Neighborhood:(str)CollgCr
HouseStyle:(str)1Story
YearRemodAdd:(int)1965
Foundation:(str)CBlock
TotRmsAbvGrd:(int)6
```

## 输出

`train log rmse0.322268`


查看`ML_task1/results/sbmission.csv`文件中内容如下
```
Id	Saleprice
1463	169819.95
```

## 改进空间
增加特征值（其实可以尝试用上所有的特征值），目前的误差值别人可以做到`0.17`左右，相比之下此模型还有优化空间。






