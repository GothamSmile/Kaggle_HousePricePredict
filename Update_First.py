import warnings

import matplotlib.pyplot as plt  # 绘图函数
import numpy as np
import pandas as pd
import seaborn as sns  # 基于matplotlib的可视化库
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings('ignore')

train = pd.read_csv(r'E:\python数据\House-Prices\train.csv')
test = pd.read_csv(r'E:\python数据\House-Prices\test.csv')
train_ID = train['Id']
test_ID = test['Id']  # 两表的Id数目不一致
# train.drop("Id", axis=1, inplace=True)
# test.drop("Id", axis=1, inplace=True)
train.columns
'''
Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice'],
      dtype='object')
'''
test.columns
'''
Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition'],
      dtype='object')
'''
train['SalePrice'].describe()
'''
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: SalePrice, dtype: float64
'''
# 查看SalePrice(目标值)是否是正态分布
sns.distplot(train['SalePrice'])
'''
结果是偏态分布
'''
# skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
'''
Skewness(偏度): 1.882876
Kurtosis(峰度): 6.536282
'''
# 去掉那个大于4000的离群值的那行
train = train[train['1stFlrSF'] < 4000]
train_SalePrice = train['SalePrice']
# 扔掉合成表中的ID方便作图
all_data = pd.concat((train, test)).reset_index(drop=True)
# all_data.drop(['Id'], axis=1, inplace=True)
# all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.drop(['Id'], axis=1, inplace=True)
print(train.head(5))
print(all_data.shape)
print(train_SalePrice)
print("all_data size is : {}".format(all_data.shape))
# train_SalePrice.isnull()
'''
all_data size is :     (2918,79)
'''

'''
数据初步处理
观察载入数据的基本情况
'''
print("数据共有", all_data.shape[1], "列,", all_data.shape[0], "行\n")
print("训练集有", train.shape[0], '行', "测试集有", test.shape[0], "行\n")
print("训练集有", train.shape[1], '列', "测试集有", test.shape[1], "列\n")
print("object数据共有", all_data.dtypes[all_data.dtypes == 'object'].value_counts().sum(), "列\n")
print("非object数据共有", all_data.dtypes[all_data.dtypes != 'object'].value_counts().sum(), "列\n")

'''
数据共有 79+(1) 列, 2919-1(离群值) 行
训练集有 1460-1（离群值） 行 测试集有 1459 行
训练集有 80 列 测试集有 80 列
object数据共有 43 列
非object数据共有 37列   
'''

print("有", all_data.isnull().sum()[all_data.isnull().sum() > 0].shape[0], "列数据出现缺失")

print(pd.DataFrame({"missing ratio": all_data.isnull().sum()[all_data.isnull().sum() > 0]
                   .sort_values(ascending=False) / all_data.shape[0]}))

'''
有 34+(1)（SalePrice） 列数据出现缺失
              missing ratio
PoolQC             0.996574
MiscFeature        0.964029
Alley              0.932169
Fence              0.804385
#SalePrice          0.499829
FireplaceQu        0.486468
LotFrontage        0.166495
GarageYrBlt        0.054471
GarageFinish       0.054471
GarageQual         0.054471
GarageCond         0.054471
GarageType         0.053786
BsmtCond           0.028092
BsmtExposure       0.028092
BsmtQual           0.027749
BsmtFinType2       0.027407
BsmtFinType1       0.027064
MasVnrType         0.008222
MasVnrArea         0.007879
MSZoning           0.001370
#这里都是一样的空值状态（1个2个）
BsmtFullBath       0.000685
BsmtHalfBath       0.000685
Functional         0.000685
Utilities          0.000685
BsmtFinSF2         0.000343
BsmtUnfSF          0.000343
BsmtFinSF1         0.000343
TotalBsmtSF        0.000343
SaleType           0.000343
KitchenQual        0.000343
Exterior2nd        0.000343
Exterior1st        0.000343
GarageCars         0.000343
GarageArea         0.000343
Electrical         0.000343

PoolQC 游泳池情况，取值有Excellent，Good，Fair，空值表示没有游泳池。
MiscFeature 房屋有的其他设施，取值有Shed(小屋)，Othr(其他)，Gar2(第二个车库)，TenC(乒乓球桌)，空值代表没有其他设施
Alley 小巷道路类型，取值有Grvl（碎石），paved(铺切面)。空值代表没有小巷
Fence 围墙类型，取值有GdPrv(Good Privacy)，MnPrv（Minimum Privacy），GdWo（Good Wood），MnWw（Minimum Wood），空值代表没有围墙
FireplaceQu 壁炉类型，取值有Ex(Excellent),Gd(Good),Fa(Fair),Po(Poor),TA(Average),空值代表没有壁炉。
SalePrice例外
3列数据缺失值接近100%,2列接近50%
'''
# 所以讲这些列都赋予空值，赋予None字符串
# 先解决严重缺失的列
all_data['PoolQC'].fillna("None", inplace=True)
all_data['MiscFeature'].fillna("None", inplace=True)
all_data['Alley'].fillna("None", inplace=True)
all_data['Fence'].fillna("None", inplace=True)
# all_data['SalePrice'].fillna(all_data['SalePrice'].mean(), inplace=True)
all_data['FireplaceQu'].fillna("None", inplace=True)
# 其余那些没那么严重缺失的列19-5=14列
all_data['LotFrontage'] = all_data.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_data['GarageQual'].fillna("None", inplace=True)
all_data['GarageCond'].fillna("None", inplace=True)
all_data['GarageArea'].fillna(0, inplace=True)
all_data['GarageCars'].fillna(0, inplace=True)
all_data['GarageFinish'].fillna("None", inplace=True)
all_data['GarageYrBlt'].fillna(0, inplace=True)
all_data['GarageType'].fillna("None", inplace=True)
all_data['BsmtQual'].fillna("None", inplace=True)
all_data['BsmtExposure'].fillna("None", inplace=True)
all_data['BsmtCond'].fillna("None", inplace=True)
all_data['BsmtFinType1'].fillna("None", inplace=True)
all_data['BsmtFinType2'].fillna("None", inplace=True)
all_data['BsmtFullBath'].fillna("None", inplace=True)
all_data['BsmtHalfBath'].fillna("None", inplace=True)
all_data['Functional'].fillna("Typ", inplace=True)
all_data['Utilities'].fillna("None", inplace=True)
all_data['BsmtFinSF2'].fillna("None", inplace=True)
all_data['BsmtUnfSF'].fillna("None", inplace=True)
all_data['BsmtFinSF1'].fillna("None", inplace=True)
all_data['TotalBsmtSF'].fillna("None", inplace=True)
all_data['SaleType'].fillna("WD", inplace=True)  # 只有一个空值
all_data['MasVnrType'].fillna("None", inplace=True)
all_data['MSZoning'].fillna("RL", inplace=True)
all_data['MasVnrArea'].fillna(0, inplace=True)
all_data['Electrical'].fillna("None", inplace=True)
all_data['Exterior1st'].fillna("None", inplace=True)
all_data['Exterior2nd'].fillna("None", inplace=True)
all_data['KitchenQual'].fillna("TA", inplace=True)  # 只有一个空值

# 另外某些特征值是数字，但它并不是连续型数据，而是离散型的，将它们转换成文本格式。
'''
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
'''
# all_data = all_data.drop(['Utilities'], axis=1)
'''
初步数据处理基本处理完毕
开始进行数据可视化
'''
train = all_data[:train.shape[0]]
# train = pd.concat((train, train_SalePrice), axis=1)

# 首先看看SalePrice的分布情况
# all_data.head(5)
# fig = plt.figure(figsize=(14, 8))
# plt.subplot2grid((2, 2), (0, 0))
# sns.distplot((train['SalePrice']))
# plt.subplot2grid((2, 2), (0, 1))
# sns.distplot(np.log((train['SalePrice'])), fit=norm)
# plt.subplot2grid((2, 2), (1, 0))
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.subplot2grid((2, 2), (1, 1))
# res = stats.probplot(np.log(train['SalePrice']), plot=plt)
#
# # 查看各变量关联性的热力图变量与标签SalePrice的关联分布
# corrmat = abs(train.corr())
# plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
#
# # 变量与标签SalePrice的关联分布
# fig = plt.figure(figsize=(14, 8))
# abs(train.corr()['SalePrice']).sort_values(ascending=True).plot.bar()
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
'''
可以看到有十多个变量与标签的相关系数是大于0.5。如
1.OverallQual 房屋设施的完整性
2.GrLivArea 居住面积
3.4 GarageCars GarageArea 车库容量，看来老外很看重车库，因为人人有车
5.TotalBsmtSF 地下室大小
6.1stFlrSF 1楼面积
7.FullBath 厕所面积
8.TotRmsAbvGrd 地面上的房间数量
9.YearBuilt 建造年份
10.YearRemodAdd 重建年份
'''

# 先看OverallQual 的情况
# fig = plt.figure(figsize=(10, 6))
# fig = sns.boxplot(x="OverallQual", y="SalePrice", data=train[['SalePrice', 'OverallQual']])
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("OverallQual", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
# # 从上图很容易可以看出，OverallQual越高，房屋价格越高。
#
# # GrLivArea居住面积与销售价格的关系
# fig = plt.figure(figsize=(10, 6))
# plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("GrLivArea", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
# plt.show()
# '''上图可以看到，居住面积与销售价格具有很强烈的正相关。但是样本中也有两个右下角离群点，
# 这两个离群点可能会对数据拟合不利，所以有些人会把这两个点去掉，
# 在最后的处理，会尝试查看这两个点去掉与否对误差的影响。
# '''
# # 查看车库容量和面积对销售价格的影响
#
#
# # fig = plt.figure(figsize=(20,10))
# plt.figure(figsize=(10, 20))
# # plt.subplots_adjust(wspace =0.5, hspace =0.5)#调整子图间距
# plt.subplot(311)
#
# plt.scatter(x=train['GarageArea'], y=train['SalePrice'])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("GarageArea", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
#
# plt.subplot(312)
# fig1 = sns.boxplot(x='GarageCars', y="GarageArea", data=train[['GarageArea', 'GarageCars']])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("GarageCars", fontsize=20)
# plt.ylabel("GarageArea", fontsize=20)
#
# plt.subplot(313)
# fig = sns.boxplot(x='GarageCars', y="SalePrice", data=train[['SalePrice', 'GarageCars']])
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("GarageCars", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
# plt.show()
# # 下边查看地下室面积、1楼面积、厕所数量，房间数量如销售价格的关系
# # fig = plt.figure(figsize=(20,10))
# plt.figure(figsize=(10, 25))
# # plt.subplots_adjust(wspace =0.5, hspace =0.5)#调整子图间距
# plt.subplot(411)
#
# plt.scatter(x=train['TotalBsmtSF'], y=train['SalePrice'])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("TotalBsmtSF", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
#
# plt.subplot(412)
#
# plt.scatter(x=train['1stFlrSF'], y=train['SalePrice'])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("1stFlrSF", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
#
# plt.subplot(413)
# fig = sns.boxplot(x='FullBath', y="SalePrice", data=train[['SalePrice', 'FullBath']])
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("FullBath", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
#
# plt.subplot(414)
# fig = sns.boxplot(x='TotRmsAbvGrd', y="SalePrice", data=train[['SalePrice', 'TotRmsAbvGrd']])
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("TotRmsAbvGrd", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
# plt.show()
'''
从两个面积的散点图可以看到，地下室和地上面积越大，销售价格越高。而且都存在数个离群点，
特别是有一个面积特别大但是销售价格只有十多万的一个数据，同一个异常数据在很多列都出现了，
这对最后的拟合很大可能会造成影响，因此我们决定还是把它去掉。
'''

# 建造年份和维修年份对价格的影响
# fig = plt.figure(figsize=(20,10))
# plt.figure(figsize=(20, 5))
# #plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距
# plt.subplot(121)
#
# plt.scatter(x=train['YearBuilt'], y=train['SalePrice'])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("YearBuilt", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
#
# plt.subplot(122)
#
# plt.scatter(x=train['YearRemodAdd'], y=train['SalePrice'])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("YearRemodAdd", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
'''
非数值型数据的关联
'''
# 对于非数值型数据,需要将其转变为离散型变量才可以处理，所以先用sklearn里内置的模块处理一下
from sklearn.preprocessing import LabelEncoder

train_shape = train.shape[0]
# all_data = pd.concat((train, test))
cols = all_data.dtypes[all_data.dtypes == 'object'].index  # train与test组合的表中的类型为object的索引为  cols的列表

for col in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[col].values))
    all_data[col] = lbl.transform(list(all_data[col].values))
print('Shape all_data: {}'.format(all_data.shape))
'''
Shape all_data: (2919, 80)     79:没有SalePrice
'''
# 增加一个新特征总面积
# all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

train = all_data[:train_shape]
test = all_data[train_shape:]

cols = list(cols)
cols.append('SalePrice')
train.columns
train[cols].columns
# #热力图   问题：非数值型没转换成功
# corrmat = abs(train[cols].corr())
# plt.subplots(figsize=(12, 12))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# # 关联
# fig = plt.figure(figsize=(18, 10))
# abs(train[cols].corr()['SalePrice']).sort_values(ascending=True).plot.bar()
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
'''
得出相关性超过50%的只有四个
GarageFinish 
BsmtQual
KitchenQual
ExterQual
'''
# #fig = plt.figure(figsize=(20,10))
# plt.figure(figsize=(10, 15))
# plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距
# plt.subplot(311)
#
# fig = sns.boxplot(x='KitchenQual', y="SalePrice", data=train[['SalePrice', 'KitchenQual']])
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("KitchenQual", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
#
# plt.subplot(312)
#
# fig = sns.boxplot(x='BsmtQual', y="SalePrice", data=train[['SalePrice', 'BsmtQual']])
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("BsmtQual", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
#
# plt.subplot(313)
# fig = sns.boxplot(x='ExterQual', y="SalePrice", data=train[['SalePrice', 'ExterQual']])
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("ExterQual", fontsize=20)
# plt.ylabel("SalePrice", fontsize=20)
#
# plt.show()
'''
对于非数值型数据，我们可以用sklearn内置的LableEncoder来对其编码，编码后就可以观察其与标签的关系，同时也可以参与训练。

编码后发现只有3个指标与销售价格相关性大于50%，其他指标都基本低于40%
'''
'''
3.特征工程
观察数据的分布，对其有一定了解之后，需要对特征作进一步处理了。其实上边已经对一些数据进行填充空值、编码等处理。这里的话会对数据进行继续完善。

3.1进一步编码
查看发现，数据中有一些特征虽然其内容是数字，但是实际上数字大小并没有实际意义，因此可以将其编码为字符，再用LabelEndcoder来转换其编码。
这里只有一个特征是这种情况：MSSubClass。虽然OverallQual，OverallQual这两列数字（1-10代表从差到好）与其它数值变量含义如面积等变量意义不一样），
但是也不用重新编码了。因此这里只对MSubClass进行处理

'''
all_data['MSSubClass'].astype(str)
lbl = LabelEncoder()
lbl.fit(list(all_data['MSSubClass'].values))
all_data['MSSubClass'] = lbl.transform(all_data['MSSubClass'].values)
'''
3.2偏度处理
接下来是对一些偏度过大的数据作一下处理。我们认为当数据偏度为0的时候属于正态分布，这时数据用于拟合是最理想的。
查看发现scipy.special.boxcox1p这个函数可以对偏度大的数据作转换。
当λ=0，其变换就是x′=log(x),当λ=1时，就是x′=x−1 
http://onlinestatbook.com/2/transformations/box-cox.html
'''
# 首先查看一下偏度较大的数据
numeric_col = all_data.dtypes[all_data.dtypes != "object"].index
skew = pd.DataFrame({"Skew": abs(all_data[numeric_col].skew()).sort_values()})
skew_col = skew[abs(skew["Skew"]) > 0.75]["Skew"].sort_values(ascending=False).index
skew[abs(skew["Skew"]) > 0.75]["Skew"].sort_values(ascending=False)
# 可以看到大概有20列数据偏度是大于0.75的。其中有些偏度超过了10,下边先查看一下这些数据的分布。
"""output
Utilities        33.990952
MiscVal          21.958480
PoolQC           20.734650
PoolArea         16.907017
Street           15.508104
LotArea          12.829025
LowQualFinSF     12.094977
Heating          12.084999
Condition2       12.066294
3SsnPorch        11.381914
RoofMatl          8.712245
MiscFeature       5.066925
LandSlope         4.977715
BsmtHalfBath      4.415993
KitchenAbvGr      4.304467
Functional        4.057843
EnclosedPorch     4.005950
ScreenPorch       3.948723
GarageYrBlt       3.908213
SaleType          3.729821
GarageCond        3.597639
CentralAir        3.460801
BsmtFinSF2        3.408112
LandContour       3.118295
GarageQual        3.075732
Electrical        3.049382
BsmtFinType2      3.045893
Condition1        2.984648
PavedDrive        2.980616
BsmtCond          2.864057
SaleCondition     2.789472
MasVnrArea        2.614936
OpenPorchSF       2.536417
ExterCond         2.499003
BldgType          2.193388
Fence             1.994802
WoodDeckSF        1.843380
ExterQual         1.802335
MSZoning          1.752645
RoofStyle         1.554106
LotFrontage       1.506478
1stFlrSF          1.470360
KitchenQual       1.448768
GrLivArea         1.270010
LotConfig         1.196901
BsmtExposure      1.115430
SalePrice         1.099357
2ndFlrSF          0.862118
BsmtFullBath      0.788251
TotRmsAbvGrd      0.758757
Name: Skew, dtype: float64
"""
# fig = plt.figure(figsize=(20,10))
# plt.figure(figsize=(15, 10))
# plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距
#
# plt.subplot(321)
# sns.distplot(train['MiscVal'])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("MiscVal", fontsize=20)
#
# plt.subplot(322)
# sns.distplot((train['PoolArea']))
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("PoolArea", fontsize=20)
#
# plt.subplot(323)
# sns.distplot((train['LotArea']))
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("LotArea", fontsize=20)
#
# plt.subplot(324)
# sns.distplot((train['LowQualFinSF']))
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("LowQualFinSF", fontsize=20)
#
# plt.subplot(325)
# sns.distplot((train['3SsnPorch']))
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("3SsnPorch", fontsize=20)
#
# plt.subplot(326)
# sns.distplot((train['GrLivArea']))
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("GrLivArea", fontsize=20)
'''
上图画了5张偏度大于10的特征分布，以及一个偏度为1.1的特征分布。可以看到偏度很高的分布，其原因是很多值都为0，
比如PoolArea游泳池面积,LotArea花园面积，如果没有游泳池或花园，这些值就为0。对于这种0值很多的分布，boxcox1p也无法将其还原到正态分布。
而对于GrLivArea居住面积这种分布，boxcox1p是有希望将其恢复为正态分布的。下边就试一下。
'''
all_data = pd.get_dummies(all_data)
all_data.shape
"""output
1.(2919, 3770)        2. (2918, 3767)
"""
train_data = all_data[:train.shape[0]]
test_data = all_data[train.shape[0]:]

# all_data1 = all_data.copy()
from scipy.special import boxcox1p

lam = 0.

for feat in skew_col:
    all_data[feat] = boxcox1p(all_data[feat], lam)

abs(all_data[skew_col].skew()).sort_values(ascending=False)

"""output
#Utilities        32.722387
PoolQC           22.874990
Street           15.508104
PoolArea         15.006047
Heating           9.592262
3SsnPorch         8.829794
LowQualFinSF      8.562091
RoofMatl          8.254887
Functional        5.461372
MiscVal           5.216665
Condition2        4.781968
SaleType          4.651086
MiscFeature       4.499531
LandSlope         4.486542
GarageCond        4.319114
BsmtFinType2      3.951148
GarageYrBlt       3.928000
BsmtHalfBath      3.855942
LandContour       3.528663
GarageQual        3.528110
KitchenAbvGr      3.522161
CentralAir        3.460801
MSZoning          3.389816
BsmtCond          3.320867
SaleCondition     3.259505
ExterCond         3.182679
Electrical        3.171859
PavedDrive        3.081842
ExterQual         3.049371
ScreenPorch       2.947420
Fence             2.669848
BsmtFinSF2        2.499116
KitchenQual       2.327632
BldgType          1.971066
EnclosedPorch     1.962089
BsmtExposure      1.457024
LotConfig         1.353095
RoofStyle         1.294681
LotFrontage       1.024132
Condition1        0.959142
MasVnrArea        0.537294
LotArea           0.505010
BsmtFullBath      0.446063
2ndFlrSF          0.305206
WoodDeckSF        0.158114
1stFlrSF          0.064861
OpenPorchSF       0.041819
TotRmsAbvGrd      0.035125
GrLivArea         0.013194
SalePrice         0.005933
"""
for col in cols:
    if col == 'SalePrice': continue
    all_data[col] = all_data[col].apply(str)




test_data.isnull
train_SalePrice = np.log1p(train_data['SalePrice'])
del test_data['SalePrice']
del train_data['SalePrice']

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

n_folds = 5


def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_data.values)
    rmse = np.sqrt(
        -cross_val_score(model, train_data.values, train_SalePrice.values, scoring="neg_mean_squared_error", cv=kf))
    return rmse


# train_SalePrice.fillna(0, inplace=True)
train_SalePrice.values

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ridge = make_pipeline(RobustScaler(), Ridge(alpha=0.0005, random_state=1))

print(np.isnan(train_SalePrice).any())
print(np.isnan(train_data).any())
print(np.isfinite(train_data).all())
print(np.isfinite(train_SalePrice).all())

score = rmse_cv(lasso)
print("Lasso Score: ", score.mean())
score = rmse_cv(ridge)
print("ridge Score: ", score.mean())
"""output
Lasso Score:  0.01034982198351518   (2020-7-7-22:52)
ridge Score:  0.015585908771924423
"""

KRR = KernelRidge()
score = rmse_cv(KRR)
print("KRR Score: ", score.mean())
"""output
KRR Score:  0.009963253857613357   (2020-7-7-22:52)

"""

GB = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                               max_depth=4, max_features='sqrt',
                               min_samples_leaf=15, min_samples_split=10,
                               loss='huber', random_state=5)

score = rmse_cv(GB)
print("GradientBoostingRegressor Score: ", score.mean())
"""output
GradientBoostingRegressor Score:  0.009586011825112673   (2020-7-7-22:52)
"""

xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                       learning_rate=0.05, max_depth=3,
                       min_child_weight=1.7817, n_estimators=2200,
                       reg_alpha=0.4640, reg_lambda=0.8571,
                       subsample=0.5213, silent=1,
                       random_state=7, nthread=-1)

score = rmse_cv(xgb)
print("XGB Score: ", score.mean())
"""output
XGB Score:   0.019378119396264736   (2020-7-7-22:52)
"""

model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)  # ,loss='huber')
score = rmse_cv(model_lgb)
print("LGBM score:", score.mean())

"""output
LGBM score: 0.009506583433376386    (2020-7-7-22:52)
"""


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        # print(self.base_models_)
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        # print(self.meta_model)
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                # print("before fit")
                # print(train_index)
                instance.fit(X[train_index], y[train_index])
                # print("after fit")
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


stacked_averaged_models = StackingAveragedModels(base_models=(lasso, model_lgb, KRR, ridge, GB, xgb),
                                                 meta_model=lasso)

score = rmse_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

"""output
Stacking Averaged models score: 0.0090 (0.0009)   (2020-7-7-22:52)

"""
stacked_averaged_models.fit(train_data.values, train_SalePrice.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test_data.values))
result = pd.DataFrame({'Id': test_ID, 'SalePrice': stacked_pred})

result.to_csv(r'E:\python数据\House-Prices\submission_4.csv', index=False)

train_data.columns
