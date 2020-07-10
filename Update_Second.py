import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
import warnings


def ignore_warn(*args, **kwargs):
    pass
    warnings.warn = ignore_warn  # ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew  # for some statistics

# 读取数据
train = pd.read_csv(r'E:\python数据\House-Prices\train.csv')
test = pd.read_csv(r'E:\python数据\House-Prices\test.csv')
train_ID = train['Id']
test_ID = test['Id']  # 两表的Id数目不一致

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)
# 检查样本和特征的数量
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))
'''
进入数据可视化与数据处理（含数据清洗以及正则化）环节
'''
fig, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
# 看见有两个离群值，去除
# Deleting outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

sns.distplot(train['SalePrice'], fit=norm);
# 分析SalePrice的情况
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])  # 求平均数以及标准差
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))  # 保留两位小数的格式
'''
mu = 180932.92 and sigma = 79467.79
目标变量向右倾斜。由于（线性）模型喜欢正态分布的数据，我们需要转换这个变量，使其更为正态分布。
'''
# Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])  # 显示对数形式

# Check the new distribution
sns.distplot(train['SalePrice'], fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
'''
mu = 12.02 and sigma = 0.40
'''
# Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
'''
进入特征工程
'''

# 连接训练集train以及测试集test
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)  #去掉Id索引
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
'''
all_data size is : (2917, 79)
'''
# 查看缺失值
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data
# 查看缺失值特征
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

# Correlation map to see how features are correlated with SalePrice
# 查看缺失值热力密度图
corrmat = train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cbar=True, vmax=0.9, square=True, fmt='.2f', cmap='YlGnBu', annot_kws={'size': 10})
# 处理缺失值
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

# Check remaining missing values if any
# 处理后检查
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head()

# MSSubClass=The building class
# 转换一些真正分类的数值变量
'''
进一步编码
查看发现，数据中有一些特征虽然其内容是数字，但是实际上数字大小并没有实际意义，因此可以将其编码为字符，再用LabelEndcoder来转换其编码。
这里只有一个特征是这种情况：MSSubClass。虽然OverallQual，OverallQual这两列数字（1-10代表从差到好）与其它数值变量含义如面积等变量意义不一样），
'''
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

# Year and month sold are transformed into categorical features.
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
all_data size is : (2917, 79)
'''
# Adding total sqfootage feature
# 添加一个更重要的特征
# 由于区域相关特征对房价的决定非常重要，我们又增加了一个特征，即每套房子的地下室、一楼和二楼的总面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# 倾斜特征
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness
'''
（高度）倾斜特征的Box-Cox变换
使用scipy函数boxcox1p计算1+x的Box-Cox变换。
请注意，设置λ=0相当于上述用于目标变量的log1p。
'''
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
'''
There are 59 skewed numerical features to Box Cox transform
'''
from scipy.special import boxcox1p

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    # all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)

# all_data[skewed_features] = np.log1p(all_data[skewed_features])
# 获取虚拟分类特征
all_data = pd.get_dummies(all_data)
print(all_data.shape)

'''
(2917, 220)
'''
# 覆盖记录
train = all_data[:ntrain]
test = all_data[ntrain:]
'''
进入建模阶段
'''
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

'''
定义交叉验证策略

使用Sklearn的cross-valu-score函数。但是这个函数没有shuffle属性，我们添加一行代码，以便在交叉验证之前对数据集进行shuffle
'''
# Validation function
n_folds = 5

# 均方根差
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

# LASSO回归：这个模型可能对异常值非常敏感。所以我们需要让它对他们更加有力。为此，使用sklearn的Robustscaler（）方法
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
# Elastic Net Regression（弹性网回归）：再次对异常值保持稳健
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
# Kernel Ridge Regression（核岭回归）
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# Gradient Boosting Regression （梯度增强回归）：
# 由于huber损失使得它对异常值很稳健:
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
# XGBoost :
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)
# LightGBM :
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
'''
通过评估交叉验证rmsle错误来了解这些基本模型是如何对执行数据
'''
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
'''
Lasso score: 0.1115 (0.0074)
'''
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
'''
ElasticNet score: 0.1116 (0.0074)
'''
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
'''
Kernel Ridge score: 0.1153 (0.0075)
'''
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
'''
Gradient Boosting score: 0.1167 (0.0083)
'''
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
'''
Xgboost score: 0.1150 (0.0066)
'''
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
'''
LGBM score: 0.1165 (0.0066)
'''

'''
叠加模型
最简单的叠加方法：平均基本模型
从平均基本模型的简单方法开始。构建了一个新的类，用我们的模型扩展scikit-learn，还扩展了laverage封装和代码重用（继承）
'''
# 平均基本模型类
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

# 在这平均了四个模型，ENet, GBoost, KRR 和lasso。当然，也可以添加更多的模型。
averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
'''
Averaged base models score: 0.1087 (0.0077)
'''

'''
不那么简单的叠加：添加元模型
在这种方法中，我们在平均基本模型上添加一个元模型，并使用这些基本模型的折叠预测来训练我们的元模型。

训练部分的程序可描述如下：

将整个训练集分成两个不相交的集（这里是train和holdout）
在第一部分（train）上训练几个基本模型
在第二部分测试这些基本模型（holdout）
使用来自第三步的预测（称为折叠预测）作为输入，并使用正确的响应（目标变量）作为输出，以训练称为元模型的高级学习者。
前三步是迭代完成的。以5倍叠加为例，首先将训练数据分成5倍。然后进行5次迭代。在每次迭代中，将每个基本模型训练为4个折叠，并预测剩余折叠（保持折叠）。

因此，在5次迭代之后，整个数据将被用于获得折叠预测，然后将在步骤4中使用这些预测作为新特性来训练元模型。

在预测部分，根据测试数据对所有基本模型的预测进行平均，并将其作为元特征，在此基础上利用元模型进行最终预测。
'''
# 叠加平均模型类
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
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

# 为了使这两种方法具有可比性（通过使用相同数量的模型），只需平均ENet KRR和Gboost，然后添加lasso作为元模型。
stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                 meta_model=lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

# StackedRegressor、XGBoost和LightGBM，将XGBoost和LightGBM添加到前面定义的StackedRegressor中。
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

'''
最后训练和预测
'''
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))
'''
0.07839506096665516
'''
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
'''
0.07885804276189369
'''
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
'''
0.0725454031865089
'''


print('RMSLE score on train data:')
print(rmsle(y_train, stacked_train_pred * 0.70 +
            xgb_train_pred * 0.15 + lgb_train_pred * 0.15))
'''
RMSLE score on train data:
0.07562564172886471
'''
# 集合预测:
ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv(r'E:\python数据\House-Prices\Test11.csv', index=False)
