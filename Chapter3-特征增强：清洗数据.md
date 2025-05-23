### 3.1.2 探索性数据分析
#### 检测缺失值
* `df.info()`
* `df.isnull().sum()`
### 3.2.1 删除有害的行
#### 空准确率
* 假设原始数据集中，糖尿病患者占比65%，非糖尿病患者占比35%，空准确率指的是，总是无脑预测占比多的一类的预测模型的准确率，即65%
#### KNN和网格搜索
* 实例化一个KNN类，设置多个n_neighbers，然后网格搜索准确度最高的n_neighbers的准确度及邻居个数
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
```
### 3.2.2 填充缺失值
#### 使用sk-learn内置功能填充缺失值
* `from sklearn.preprocessing import Imputer`
### 3.2.3 在机器学习流水线中填充值
#### train, test split and preprocessing
* 错误的预处理做法：在划分训练集、测试集之前，对整个数据集进行预处理（如填充null值，标准化）
  - 这样预处理，如用整个数据集的均值填充null值，用到了测试集的信息，用到了未来数据。模型预测准确率会虚高。
* 正确的预处理做法：以填充null值为例，先计算出训练集的均值，然后用它填充测试集的缺失值
  - 这样预处理，模型预测结果准确率会比前者低，但更真实。
* train, test split需要用到同一个random state，保证可重复性
#### 使用scikit-learn的Pipeline
* `from sklearn.pipeline import Pipeline`
## 3.3 标准化和归一化
* pandas.plot()调用sharex，使得多图在同一x轴坐标下绘图
  + `df.hist(sharex=True)`
### 3.3.1 z分数标准化
* z分数标准化的输出会被重新缩放，使均值为0、标准差为1。通过缩放特征、统一化均值和方差（标准差的平方），可以让KNN这种模型达到最优化，而不会倾向于较大比例的特征。
* 标准化方法：(元素-均值)/标准差
* z分数标准化的结果是数据在x轴上更紧密了，在y轴无变化，数据的直方图形状无变化
### 3.3.2 min-max标准化
* 标准化方法：(元素-min)/(max-min)
* 标准化结果是最小值都是0，最大值都是1。这种缩放的副作用是标准差都非常小。这有可能不利于某些模型，因为异常值的权重降低了。
### 3.3.3 行归一化(?)
* 行归一化不是计算每列的统计值（均值、最小值、最大值等），而是会保证每行有单位范数（unit norm），意味着每行的向量长度相同。
* 如果每行数据都在一个n维空间内，那么每行都有一个向量范数（长度）。也就是说，我们认为每行都是空间内的一个向量
* L2范数为例，是对每行每个元素平方后，加起来求平均值
