import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

from scipy import stats
import warnings

from sklearn.kernel_ridge import KernelRidge

warnings.filterwarnings('ignore')

df_train = pd.read_csv(r'E:\python数据\House-Prices\train.csv')
test = pd.read_csv(r'E:\python数据\House-Prices\test.csv')

df_train.columns

df_train['SalePrice'].describe()

# sns.distplot(df_train['SalePrice']);
# # skewness and kurtosis
# print("Skewness: %f" % df_train['SalePrice'].skew())
# print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#
# # 居住面积平方英尺
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));
#
# # 地下室面积平方英尺
# var = 'TotalBsmtSF'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));
#
# # 整体材料和饰面质量
# var = 'OverallQual'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000);
#
# # 原施工日期
# var = 'YearBuilt'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(16, 8))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000);
# plt.xticks(rotation=90);
#
# var = 'Neighborhood'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# # fig.axis(ymin=0, ymax=800000);
# plt.xticks(rotation=90);
#
# k = 10
# corrmat = df_train.corr()
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
#                  xticklabels=cols.values, cmap='YlGnBu')
# plt.show()
#
# # scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], size=2.5)
# plt.show();
#
# # 转换前的数据分布：
# sns.distplot(df_train['SalePrice'], fit=norm);
#
# (mu, sigma) = norm.fit(df_train['SalePrice'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# # 分布图
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#            loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
#
# # QQ图
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()
#
# # 转换后的数据分布：
# # 对数变换log(1+x)
# df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
#
# # 看看新的分布
# sns.distplot(df_train['SalePrice'], fit=norm);
#
# # 参数
# (mu, sigma) = norm.fit(df_train['SalePrice'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# # 画图
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#            loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
#
# # QQ图
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()

# missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

train = pd.read_csv(r'E:\python数据\House-Prices\train.csv')
test = pd.read_csv(r'E:\python数据\House-Prices\test.csv')

# 看看数据多大的
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

# ID先留着，暂时不用
train_ID = train['Id']
test_ID = test['Id']

# 去掉ID
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)
# 发现离群点
fig, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# 去掉离群点
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
# 缺失值处理
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head(20)

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
# 没有特征。。。
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
# 通道的入口
all_data["Alley"] = all_data["Alley"].fillna("None")
# 栅栏
all_data["Fence"] = all_data["Fence"].fillna("None")
# 壁炉
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
# 到街道的距离
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# 车库的事
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
# 地下室的事
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
# 砌体
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
# 一般分区分类，用众数来吧
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
# Functional家庭功能评定
all_data["Functional"] = all_data["Functional"].fillna("Typ")
# 电力系统
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
# 厨房的品质
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
# 外部
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
# 销售类型
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
# 建筑类型
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data = all_data.drop(['Utilities'], axis=1)

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head()

# 另外某些特征值是数字，但它并不是连续型数据，而是离散型的，将它们转换成文本格式。
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape
print('Shape all_data: {}'.format(all_data.shape))
'''
Shape all_data: (2917, 78)
'''
# 增加一个新特征总面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

from scipy.stats import norm, skew

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness

'''
                   Skew
                    Skew
MiscVal        21.939672
PoolArea       17.688664
LotArea        13.109495
LowQualFinSF   12.084539
3SsnPorch      11.372080
LandSlope       4.973254
KitchenAbvGr    4.300550
BsmtFinSF2      4.144503
EnclosedPorch   4.002344
ScreenPorch     3.945101
BsmtHalfBath    3.929996
MasVnrArea      2.621719
OpenPorchSF     2.529358
WoodDeckSF      1.844792
1stFlrSF        1.257286
LotFrontage     1.103039
GrLivArea       1.068750
TotalSF         1.009157
BsmtFinSF1      0.980645
BsmtUnfSF       0.919688
2ndFlrSF        0.861556
#大于   0.75的  （四分线）
TotRmsAbvGrd    0.749232
Fireplaces      0.725278
HalfBath        0.696666
TotalBsmtSF     0.671751
BsmtFullBath    0.622415
OverallCond     0.569314
HeatingQC       0.485534
FireplaceQu     0.332611
BedroomAbvGr    0.326568
GarageArea      0.216857
OverallQual     0.189591
FullBath        0.165514
MSSubClass      0.139709
YrSold          0.131996
BsmtFinType1    0.083641
GarageCars     -0.219297
YearRemodAdd   -0.450134
BsmtQual       -0.488362
YearBuilt      -0.599194
GarageFinish   -0.609953
LotShape       -0.618564
MoSold         -0.646173
Alley          -0.651705
BsmtExposure   -1.117321
KitchenQual    -1.449814
ExterQual      -1.800062
Fence          -1.992649
ExterCond      -2.496489
BsmtCond       -2.861271
PavedDrive     -2.977741
BsmtFinType2   -3.042979
GarageQual     -3.072788
CentralAir     -3.457555
GarageCond     -3.594290
GarageYrBlt    -3.904632
Functional     -4.054126
Street        -15.494756
PoolQC        -21.217600
'''
# 选择偏度大于0.75的特征值通过scipy.special 的 boxcox1p进行转换
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# 然后通过pd.get_dummies将数据转换成one-hot编码。
all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test = all_data[ntrain:]

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             nthread=-1)
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

averaged_models.fit(train,y_train)
pred = np.exp(averaged_models.predict(test))
result=pd.DataFrame({'Id':test_ID,'SalePrice':pred})
result.to_csv(r'E:\python数据\House-Prices\submission_2.csv',index=False)
