## 5.3 特征选择的类型
### 5.3.1 基于统计的特征选择
#### 使用皮尔逊相关系数
* 皮尔逊相关系数要求每列是正态分布的（我们没有这样假设）。在很大程度上，我们也可以忽略这个要求，因为数据集很大（超过500的阈值）
* 代码示例：
```python
from sklearn.base import TransformerMixin, BaseEstimator

class CustomCorrelationChooser(TransformerMixin, BaseEstimator):
    def __init__(self, response, cols_to_keep=[], threshold=None):
        # 保存响应变量
        self.response = response
        # 保存阈值
        self.threshold = threshold
        # 初始化一个变量，存放要保留的特征名
        self.cols_to_keep = cols_to_keep

    def transform(self, X):
        # 转换会选择合适的列
        return X[self.cols_to_keep]

    def fit(self, X, *_):
        # 创建新的DataFrame，存放特征和响应
        df = pd.concat([X, self.response], axis=1)
        # 保存高于阈值的列的名称
        self.cols_to_keep = df.columns[df.corr()[df.columns[-1]].abs() > self.threshold]
        # 只保留X的列，删掉响应变量
        self.cols_to_keep = [c for c in self.cols_to_keep if c in X.columns]
        return self

# 实例化特征选择器
ccc = CustomCorrelationChooser(threshold=.2, response=y)
ccc.fit(X)

ccc.cols_to_keep

['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5']
```
#### 使用假设检验
* 在特征选择中，假设测试的原则是：“特征与响应变量没有关系”（零假设）为真还是假。我们需要在每个特征上进行检验，并决定其与响应变量是否有显著关系。
在某种程度上说，我们的相关性检测逻辑也是这样运作的。我们的意思是，如果某个特征与响应变量的相关性太弱，那么认为“特征与响应变量没有关系”这个假设为真。
如果相关系数足够强，那么拒绝该假设，认为特征与响应变量有关。
* 需要注意的是，f_classif函数在每个特征上单独（单变量测试由此得名）执行一次ANOVA测试（一种假设检验类型），并分配一个p值。SelectKBest会将特征
按p值排列（越小越好），只保留我们指定的k个最佳特征。
    + p值不是越小越好，且不能互相比较。
    + p值的一个常见阈值是0.05，意思是可以认为p值小于0.05的特征是显著的。
    + 在ANOVA之外，还有其他的测试能用于回归任务，例如卡方检验等。f_classif可以使用负数，但不是所有类都支持，chi2（卡方）也很常用，但只支持正数。
### 5.3.2 基于模型的特征选择
#### 使用机器学习选择特征
##### 特征选择指标——针对基于树的模型
* 在拟合决策树时，决策树会从根节点开始，在每个节点处贪婪地选择最优分割，优化节点纯净度指标。默认情况下，scikit-learn每步都会优化基尼指数
（gini metric）。每次分割时，模型会记录每个分割对整体优化目标的帮助。因此，在树形结构中，这些指标对特征重要性有作用。
* 之前我们使用scikit-learn内置的包装器SelectKBest，基于排序函数（例如ANOVA的p值）取前k个特征。下面会引入一个新的包装器SelectFromModel，
和SelectKBest一样选取最重要的前k个特征。但是，它会使用机器学习模型的内部指标来评估特征的重要性，不使用统计测试的p值。
    + SelectFromModel和SelectKBest相比最大的不同之处在于不使用k（需要保留的特征数）：SelectFromModel使用阈值，代表重要性的最低限度。
* 代码示例：
```python
# 为后面加速
tree_pipe_params = {'classifier__max_depth': [1, 3, 5, 7]}

# 和SelectKBest相似，但不是统计测试
from sklearn.feature_selection import SelectFromModel
# 实例化一个类，按照决策树分类器的内部指标排序重要性，选择特征
select_from_model = SelectFromModel(DecisionTreeClassifier(), threshold=.05)

from sklearn.pipeline import Pipeline

# 创建基于DecisionTreeClassifier的SelectFromModel
select = SelectFromModel(DecisionTreeClassifier())

select_from_pipe = Pipeline([('select', select), ('classifier', d_tree)])

select_from_pipe_params = deepcopy(tree_pipe_params)

# 我们可以用一些保留字作为阈值参数的一部分，并不是必须选择表示最低重要性的浮点数。例如，mean的阈值只选择比均值更重要的特征，median的阈值只选择比
# 中位数更重要的特征。我们还可以用这些保留字的倍数，例如2.*mean代表比均值重要两倍的特征。
select_from_pipe_params.update({
  'select__threshold': [.01, .05, .1, .2, .25, .3, .4, .5, .6, "mean", "median", "2.*mean"],
  'select__estimator__max_depth': [None, 1, 3, 5, 7]
  })

print(select_from_pipe_params)   
# {'select__threshold': [0.01, 0.05, 0.1, 'mean', 'median', '2.*mean'], 
# 'select__estimator__max_depth': [None, 1, 3, 5, 7], 'classifier__max_depth': [1, 3, 5, 7]}
```
##### 线性模型和正则化
* SelectFromModel可以处理任何包括feature_importances_或coef_属性的机器学习模型。基于树的模型会暴露前者，线性模型则会暴露后者。在拟合后，
线性回归、逻辑回归、支持向量机（SVM，support vector machine）等线性模型会将一个系数放在特征的斜率（重要性）前面。SelectFromModel会认为这个
系数等同于重要性，并根据拟合时的系数选择特征。
* 正则化也有助于解决多重共线性的问题，也就是说，数据中有多个线性相关的特征。L1惩罚可以强制其他线性相关特征的系数为0，保证选择器不会选择这些线性相关
的特征，有助于解决过拟合问题。
    + 但lasso有个问题，如果两个特征高度相关，如EP（市盈率的倒数）, BP（市净率的倒数）因子，lasso回归被删除的因子不确定，有可能是其中一个，有可能
    是另外一个
###### 逻辑回归
```python
# 用正则化后的逻辑回归进行选择
logistic_selector = SelectFromModel(LogisticRegression())

# 新流水线，用LogistisRegression的参数进行排列
regularization_pipe = Pipeline([('select', logistic_selector), ('classifier', tree)])

regularization_pipe_params = deepcopy(tree_pipe_params)

# L1 和L2 正则化
regularization_pipe_params.update({
  'select__threshold': [.01, .05, .1, "mean", "median", "2.*mean"],
  'select__estimator__penalty': ['l1', 'l2'],
  })

print(regularization_pipe_params)  
# {'select__threshold': [0.01, 0.05, 0.1, 'mean', 'median', '2.*mean'], 'classifier__max_depth': [1, 3, 5, 7], 
# 'select__estimator__penalty': ['l1', 'l2']}

get_best_model_and_accuracy(regularization_pipe,
  regularization_pipe_params,
  X, y)

# 比原来的好，实际上是目前最好的，也快得多

# 设置流水线最佳参数
regularization_pipe.set_params(**{'select__threshold': 0.01, 'classifier__max_depth': 5,
  'select__estimator__penalty': 'l1'})

# 拟合数据
regularization_pipe.steps[0][1].fit(X, y)
# 列出选择的列
X.columns[regularization_pipe.steps[0][1].get_support()]

Index(['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5'], dtype='object')
```
###### SVC
* 目前看来，逻辑回归分类器和支持向量分类器（SVC）的最大区别在于，后者会最大优化二分类项目的准确性，而前者对属性的建模更好。
```python
# SVC是线性模型，用线性支持在欧几里得空间内分割数据
# 只能分割二分数据
from sklearn.svm import LinearSVC

# 用SVC取参数
svc_selector = SelectFromModel(LinearSVC())

svc_pipe = Pipeline([('select', svc_selector), ('classifier', tree)])

svc_pipe_params = deepcopy(tree_pipe_params)

svc_pipe_params.update({
  'select__threshold': [.01, .05, .1, "mean", "median", "2.*mean"],
  'select__estimator__penalty': ['l1', 'l2'],
  'select__estimator__loss': ['squared_hinge', 'hinge'],
  'select__estimator__dual': [True, False]
  })

print(svc_pipe_params)   
# 'select__estimator__loss': ['squared_hinge', 'hinge'], 
# 'select__threshold': [0.01, 0.05, 0.1, 'mean', 'median', '2.*mean'], 'select__estimator__penalty': ['l1', 'l2'], 
# 'classifier__max_depth': [1, 3, 5, 7], 'select__estimator__dual': [True, False]}

get_best_model_and_accuracy(svc_pipe,
  svc_pipe_params,
  X, y)

# 刷新了纪录
```
## 5.4 选用正确的特征选择方法
* 如果特征是分类的，那么从SelectKBest开始，用卡方或基于树的选择器。
    - 分类任务用树模型
* 如果特征基本是定量的，用线性模型和基于相关性的选择器一般效果更好。
    - 回归任务用线性模型，或者基于相关性的选择器
* 如果是二元分类问题，考虑使用SelectFromModel和SVC，因为SVC会查找优化二元分类任务的系数。
    - 二元分类用SVC（有待进一步探讨）
* 在手动选择前，探索性数据分析会很有益处。不能低估领域知识的重要性。
