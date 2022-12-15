# 4.2.2 自定义分类填充器
## 用scikit-learn的TransformerMixin基类创建我们的自定义分类填充器
* `from sklearn.base import TransformerMixin`

# 4.2.3 自定义定量填充器
## 可以不分别调用并用fit_transform拟合转换CustomCategoryImputer和Custom-QuantitativeImputer，而是把它们放在流水线中
* 示例：
```
# 从sklearn导入Pipeline
from sklearn.pipeline import Pipeline
imputer = Pipeline([('quant', cqi), ('category', cci)])
imputer.fit_transform(X)
```

# 4.3.1 定类等级的编码
## 将分类数据转换为虚拟变量（dummy variable）
1. 当使用虚拟变量时，需要小心虚拟变量陷阱。虚拟变量陷阱的意思是，自变量有多重共线性或高度相关。简单地说，这些变量能依据彼此来预测。
在这个例子中，如果设置female和male两个虚拟变量，它们都可以取值为1或0，那么就出现了重复的类别，陷入了虚拟变量陷阱。我们可以直接
推断female = 0代表男性。为了避免虚拟变量陷阱，我们需要忽略一个常量或者虚拟类别。被忽略的虚拟变量可以作为基础类别，和其他变量进行比较。
2. Pandas有个很方便的get_dummies方法，可以找到所有的分类变量，并将其转换为虚拟变量
3. 另一种选择是创建一个自定义虚拟化器，从而在流水线中一口气转换整个数据集
    - 我们的自定义虚拟化器模仿了scikit-learn的OneHotEncoding，但是可以在整个DataFrame上运行

# 4.3.2 定序等级的编码
## 标签编码器
1. 在定序等级，由于数据的顺序有含义，使用虚拟变量是没有意义的。为了保持顺序，我们使用标签编码器。
2. 标签编码器是指，顺序数据的每个标签都会有一个相关数值。在我们的例子中，这意味着顺序列的值（dislike、somewhat like和like）会用0、1、2来表示。
    - `print(X['ordinal_column'].map(lambda x: ordering.index(x)))  # 将ordering映射到顺序列`
    - 注意，我们没有使用scikit-learn的LabelEncoder，因为这个方法不能像上面的代码那样对顺序进行编码（0表示dislike，1表示somewhat like，2表示like）
3. 将自定义标签编码器放进流水线中

# 4.3.3 将连续特征分箱
## 连续特征分箱
1. 有时，如果数值数据是连续的，那么将其转换为分类变量可能是有意义的。例如你的手上有年龄，但是年龄段可能会更有用。
2. Pandas有一个有用的函数叫作cut，可以将数据分箱（binning），亦称为分桶（bucketing）。
    - `pd.cut(X['quantitative_column'], bins=3)`
3. 利用cut函数的属性，可以为流水线定义自己的CustomCutter。
4. 注意，现在quantitative_column列处于定序等级，不需要引入虚拟变量。

# 4.3.4 创建流水线
1. 流程：用imputer填充缺失值，用虚拟变量填充分类列，对ordinal_column进行编码，将quantitative_column分箱。

# 4.4.1 根据胸部加速度计识别动作的数据集
1. 击败空正确率（即概率最大的预测结果的出现概率）
    - `df['target'].value_counts(normalize=True)`
    - 上述pandas方法的normalize为True表示归一化处理，即结果以百分比展示

# 4.4.2 多项式特征
## sklearn的多项式生成模块
1. 在处理数值数据、创建更多特征时，一个关键方法是使用scikit-learn的Polynomial-Features类。这个构造函数会创建新的列，它们是原有列的乘积，用于捕获特征交互。
2. 这个类会生成一个新的特征矩阵，里面是原始数据各个特征的多项式组合，阶数小于或等于指定的阶数。意思是，如果输入是二维的，
例如[a, b]，那么二阶的多项式特征就是[1, a, b, a^2, ab, b^2]。
## scikit-learn的Polynomial-Features类的参数
1. degree是多项式特征的阶数，默认值是2。
2. interaction_only是布尔值：如果为真，表示只生成互相影响/交互的特征，也就是不同阶数特征的乘积。interaction_only默认为false。
    - 个人感觉interaction_only参数使用默认false即可，因为有时候factor^2这种因子也是有意义的，如非线性市值因子
3. include_bias也是布尔值：如果为真（默认），会生成一列阶数为0的偏差列，也就是说列中全是数字1。
    - 个人感觉include_bias参数最好使用非默认的false，因阶数为0的偏差列=1，大多数情况下无意义
## scikit-learn的Polynomial-Features类也可以放在流水线中
* 示例：
```
pipe_params = {'poly_features__degree':[1, 2, 3], 'poly_features__interaction_only':
[True, False], 'classify__n_neighbors':[3, 4, 5, 6]}
```
## 加入多项式后的效果提升
1. 根据xyz三维空间的移动距离预测人体在做什么动作，加入多项式相比之前，KNN分类预测的准确率只有微小提升
2. 个人感觉提升不明显的原因：
    - 一是原始数据和预测结果的关系就比较明确，KNN网格搜索准确率达到了72%
    - 二是多项式衍生出的新特征，实际含义有限，如x^2, x*y, 现实意义不多或可以被原始特征解释

# 4.5 针对文本的特征构建
## 概述
* 我们现在进一步探索更长的文本数据。这种文本数据比单个类别的文本复杂得多，因为长文本包括一系列类别，或称为词项（token）。
* 所有的机器学习模型都需要数值输入。因此处理文本时需要有创造性，有策略地思考如何将文本数据转换为数值特征。
## 4.5.1 词袋法
1. 我们可能会将文本数据称为语料库（corpus），将语料库转换为数值表示（也就是向量化）的常见方法是词袋（bag of words），
2. 其背后的基本思想是：
    - 通过单词的出现来描述文档，完全忽略单词在文档中的位置。在它最简单的形式中，用一个袋子表示文本，不考虑语法和词序，
    - 并将这个袋子视作一个集合，其中重复度高的单词更重要。
3. 词袋的3个步骤是：
    - 分词（tokenizing）：分词过程是用空白和标点将单词分开，将其变为词项。每个可能出现的词项都有一个整数ID。
    - 计数（counting）：简单地计算文档中词项的出现次数。
    - 归一化（normalizing）：将词项在大多数文档中的重要性按逆序排列。
## 4.5.2 CountVectorizer
1. CountVectorizer是将文本数据转换为其向量表示的最常用办法。
2. 和虚拟变量类似，CountVectorizer将文本列转换为矩阵，
    - 其中的列是词项，单元值是每个文档中每个词项的出现次数。
    - 这个矩阵叫文档-词矩阵（document-term matrix），因为每行代表一个文档（在本例中是一条推文），每列代表一个词（一个单词）。
3. 代码：`from sklearn.feature_extraction.text import CountVectorizer` 
## 4.5.3 TF-IDF向量化器
1. TF-IDF向量化器由两部分组成：表示词频的TF部分，以及表示逆文档频率的IDF部分。TF-IDF是一个用于信息检索和聚类的词加权方法。
2. 对于语料库中的文档，TF-IDF会给出其中单词的权重，表示重要性。
3. TF（term frequency，词频）：衡量词在文档中出现的频率。由于文档的长度不同，词在长文中的出现次数有可能比在短文中出现的次数多得多。
因此，一般会对词频进行归一化，用其除以文档长度或文档的总词数。
4. IDF（inverse document frequency，逆文档频率）：衡量词的重要性。在计算词频时，我们认为所有的词都同等重要。但是某些词（如is、of和that）有可能出现很多次，
但这些词并不重要。因此，我们需要减少常见词的权重，加大稀有词的权重。
5. 再次强调，TfidfVectorizer和CountVectorizer相同，都从词项构造了特征，但是TfidfVectorizer进一步将词项计数按照在语料库中出现的频率进行了归一化。
6. 代码：`from sklearn.feature_extraction.text import TfidfVectorizer`
## 4.5.4 在机器学习流水线中使用文本
1. 本例要处理大量的列（数十万），所以我们使用在这种情况下更高效的分类器——朴素贝叶斯（naive Bayes）模型：
2. pipeline步骤：
    - 用CountVectorizer将推文变成特征
    - 用朴素贝叶斯模型MultiNomialNB进行正负面情绪的分类。
3. scikit-learn有一个FeatureUnion模块，可以水平（并排）排列特征。这样，在一个流水线中可以使用多种类型的文本特征构建器。
4. 值得注意的是，CountVectorizer的最佳ngram_range是(1, 2)，而TfidfVectorizer的是(1, 1)，代表单个词的出现没有2个单词的短语那么重要。
