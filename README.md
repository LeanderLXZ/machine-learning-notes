---
# Overview
##[Step 1: EDA(Exploratory Data Analysis)](#step1)

##[Step 2: Data Preprocessing](#step2)

##[Step 3: Feature Engineering](#step3)

##[Step 4: Modeling](#step4)

##[Step 5: Ensemble](#step5)

##[Others](#others)

##[Experiences](#experiences)

##[Materials](#materials)

---
<a id='step1'></a>
# Step 1: EDA(Exploratory Data Analysis)
https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-2-connect
https://www.kaggle.com/poonaml/two-sigma-renthop-eda
https://www.kaggle.com/neviadomski/data-exploration-two-sigma-renthop
[A Complete Tutorial which teaches Data Exploration in detail](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/)

## 1. Statistic Analysis
- min/max/mean/meduim/std
- ![IMAGE](resources/0CD70CE75FE7F4118B00E674CC803ED1.jpg =900x)
- Correlation Coefficient（相关系数）矩阵
- ![IMAGE](resources/13EEC95A5C65CCC353E2CD326E84FA18.jpg =900x)
- class(positive/negative)
- 通过一些统计上的测试来验证一些假设的显著性。(数据是否符合i.i.d.(独立同分布))

## 2. Visulization
- 直方(Histogram Plot)/散点(Scatter Plot)分布图
  - 查看类数据分布趋势、密度和是否有离群点
  - feature/label分布是否均衡
  - train/test分布是否一致
  - 数据是否符合i.i.d.
  - ![IMAGE](resources/0F4030639E6BD266BF3B355B65D870D0.jpg =900x)
  - ![IMAGE](resources/01102297C761364F1276CDB64CBC996B.jpg =900x)
- 箱形图(Box Plot)
  - 可以直观查看数值变量的分布
  - ![IMAGE](resources/8BAD2A2B00EED0B27064E3ADCD605D41.jpg =900x)
- 琴形图(Violin Plot)
  - 表征了在一个或多个分类变量情况下连续变量数据的分布，并进行了比较，是一种观察多个数据分布有效方法
  - ![IMAGE](resources/BE3C40A3DF25CCE4F9EAB2A19E11465F.jpg =900x)
- Correlation Coefficient图，表征变量之间两两分布和相关度
  - ![IMAGE](resources/945CAF8F9144C480568DFCD5740C6953.jpg =900x)
- 常用工具：matplotlib/seaborn

---
<a id='step2'></a>
# Step 2: Data Preprocessing

## 1. Load Source Data
  - 数据太大 -> OOM -> batch

## 2. Missing Data
- 用平均值、中值、上下数据、分位数、众数、随机值等填充
- 用其他变量做预测模型来算出缺失变量
- 把变量映射到高维空间(增加缺失维度为新特征)
- 直接舍弃feature或样本（缺失值过多）
- 忽略（一些模型，如随机森林，自身能够处理数据缺失的情况）
- 时序数据：平滑、寻找相似、用插值法进行填充

## 3. Data Cleansing
- Drop Outliers
  - 直接剔除样本
  - 将outliers值归为上下限
- 文本数据：
  - 垃圾字符、错别字(词)、数学公式、不统一单位和日期格式等
  - 处理标点符号、分词、去停用词
  - 英文文本可能还要词性还原(lemmatize)、抽取词干(stem)等。

## 4. Normalization(Scaling)
- Standard Scale
- Min-Max Scale


## 5. Convert Categorical Variables to Dummy Variables
- 一般方法：one-hot-encoding
- 类变量太多：Feature Encoding

[Simple Methods to deal with Categorical Variables in Predictive Modeling](https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/)

## 6. Data Augmentation
- 增强鲁棒性(降低过拟合)
- 扩充数据集

[Building Powerful Image Classification Models Using Very Little Data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

---
<a id='step3'></a>
# Step 3: Feature Engineering
特征没做好，调参调到老
Kaggle：Feature 为主，调参和 Ensemble 为辅

## 1. Feature Extraction
总体：尽可能多地抽取特征，只要你认为某个特征对解决问题有帮助，它就可以成为一个特征，相信 model能够挑出最有用的feature。
- 凭经验、对任务的直观判断、思考数据集构造(magic feature)
- 数值型特征：线性组合、多项式组合来构造新feature
- 文本特征：文本长度、Word Embeddings、TF-IDF、LDA、LSI等，深度学习提取feature
- 稀疏特征：特征合并后one-hot-encoding（cat allowed和dog allowed合并成为pet allowed）
- 图片特征：bottleneck feature
- 时序问题：滑窗feature
- 通过Feature Importance（Random Forest、XGBoost对每个feature在分类上面的重要程度进行评估），对重要特种再提取

## 2. Feature Selection
过多的feature会造成冗余、噪声、容易过拟合
- Correlation Coefficient（相关系数）衡量两个变量之间的线性关系，数值在[-1.0, 1.0]区间中。数值越接近0，两个变量越线性不相关。但数值为0时，并不能说明两个变量不相关，只是线性不相关而已。如果两个Feature的相关度很高，就有可能存在冗余。
- 训练模型来筛选特征，如Random Forest、GDBT、XGBoost等，看Feature Importance。Feature Importance 对于某些数据经过脱敏处理的比赛尤其重要。

## 3. Feature Encoding
High Categorical(高势集类别)，如邮编。
- 进行经验贝叶斯转换成数值feature（统计学思路：类 -> 频率）
- 根据 Feature Importance 或变量取值在数据中的出现频率，为最重要（比如说前 95% 的 Importance）的取值创建 Dummy Variables，而其他取值都归到一个“其他”类里面

## 4. Dimensionality Reduction
- 替换原有n个features
- 扩充为新feature
- 硬聚类算法：k-Means（k均值）算法、FCM算法
- 线性方法：__PCA（主成分分析）__、SVD（奇异值分解）
- 非线性方法：__t-SNE聚类__、Sammon映射、Isomap、LLE、CCA、SNE、MVU等
- 深度学习降维，如embedding、bottleneck feature、autoencoders、denoising autoencoder

---
<a id='step4'></a>
# Step 4: Modeling

## 1. Models
- Baseline Model
- Week Models
  - Linear Regression(带惩罚项)
  - Logistic Regression
  - KNN(K-Nearest Neighbor)
  - Decision Tree
  - [A Complete Tutorial on Tree Based Modeling from Scratch](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)
  - Extra Tree
  - SVM(SVC/SVR)
    - [Understanding Support Vector Machine algorithm from examples (along with code)](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)
- Strong Models
  - __Random Forest__
  - GBM(Gradient Boosting Machines)
  - __GBDT(Gradient Boosting Decision Trees)__
  - GBRT(Gradient Boosted Regression Trees)
  - AdaBoost
  - __XGBoost__
  - LightGBM
  - CatBoost
- Temporal Models
  - moving average
  - exponential smoothing
  - Markov Model/Hidden Markov Model
  - __ARIMA__
  - RNN/LSTM
- Neural Networks(Deep Learning)
## 2. Cross Validation
主要目的为评估模型、用于模型选择（包括模型种类、参数、结构）
CV分数很可能和LB分数不一致，如何选择Case By Case
- Simple Split
- k-Fold
  - 降低variance，提升模型鲁棒性，降低overfitting的风险
  - Fold越多训练也就会越慢，需要根据实际情况进行取舍
- group-k-fold(reduce overfitting)
- Adversarial Validation
  - 从train set中选出和test set中最相似的部分作为valid set
  - [Adversarial Validation](http://fastml.com/adversarial-validation-part-one/)
- 时序问题：valid在train后、滑窗

[Improve Your Model Performance using Cross Validation](https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/)

## 3. Cost Function & Evaluation Fuction
- MSE/RMSE
- L1\_Loss(误差接近0的时候不平滑)/L2\_Loss
- Hinge\_Loss/Margin\_Loss
- Cross-entropy\_Loss/Sigmoid\_Cross-ntropy\_Loss/Softmax\_Cross-entropy\_Loss
- Log\_Loss
- ROC/AUC
- 难以定义目标loss：End2End强化学习
## 4. Parameter Tuning
- Algorithms：
  - Random Search
  - Grid Search
  - TPOT(Tree-based Pipeline Optimisation Technique)，基于遗传算法自动选择、优化机器学习模型和参数
    - https://github.com/EpistasisLab/tpot
    - [Automate Your Machine Learning](https://blog.alookanalytics.com/2017/05/25/automate-your-machine-learning/)
- Steps：
  1) 根据经验，选出对模型效果影响较大的超参
  2) 按照经验设置超参的搜索空间，比如学习率的搜索空间为[0.001，0.01, 0.1]
  3) 选择搜索算法，如Grid Search、一些启发式搜索的方法
  4) 验证模型的泛化能力

[Complete Guide to Parameter Tuning in Gradient Boosting (GBM)](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
[Complete Guide to Parameter Tuning in XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

---
<a id='step5'></a>
# Step 5: Ensemble
Feature决定了模型效果的上限，而Ensemble就是让你更接近这个上限。
将多个不同的Base Model组合成一个Ensemble Model。可以同时降低最终模型的Bias和Variance，从而在提高分数的同时又降低Overfitting的风险。
Kaggle: 不用Ensemble几乎不可能得奖
- Methods：
  _Averaging_: 对每个Base Model生成的结果取（加权）平均。
  _Voting_: 对每个Base Model生成的结果进行投票，并选择最高票数的结果为最终结果。
  - __Bagging__：使用训练数据的不同随机子集来训练每个Base Model，最后进行每个Base Model权重相同的Average或Vote。
    - 多个Base Model的线性组合。
    - 也是Random Forest的原理。
    - [Quick Guide to Boosting Algorithms in Machine Learning](https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/)
  - __Boosting__：__“知错能改”__。迭代地训练也是Random Forest的原理，每次根据上一个迭代中预测错误的情况修改训练样本的权重。
    - 比Bagging效果好，但更容易Overfit。
    - 也是Gradient Boosting的原理。
  - __Blending__：用不相交的数据训练不同的Base Model，将它们的输出取（加权）平均。
    - 实现简单，但对训练数据利用少了。
  - __Stacking__：用新的Stack Model学习怎么组合那些Base Model。
    - 多个Base Model的非线性组合。
    ![IMAGE](resources/8E7F50CA710F1453C823613F6A5405AD.jpg =900x)
    - 为了避免Label Leak，需要对每个学习器使用k-Fold，将k个模型对valid set的预测结果拼起来，作为下一层学习器的输入。
    ![IMAGE](resources/0260129CB69522536EC2FB5A9FE47BFD.jpg =900x)
    - feature复用
- Notes:
  - __Base Model 之间的相关性要尽可能的小__。Ensemble 的 Diversity 越大，最终 Model 的 Bias 就越低。
  - __Base Model 之间的性能表现不能差距太大__。
  - __Trade-off__
    - [Trade-Off Between Diversity and Accuracy in Ensemble Generation](https://link.springer.com/chapter/10.1007%2F3-540-33019-4_19)

[Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/)
[Basics of Ensemble Learning Explained in Simple English](https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/)

---
<a id='others'></a>
# Others
## Other Algorithms
- Naive Bayes
- Bayes networks
- EM(Expectation Maximization Algorithm)
- Gaussian Mixture Models
- Mixture of Gaussians Clustering

## Imbalanced Data
label类别不均衡问题。
- Stratified k-Fold
- under-sampling(欠采样)
  - EasyEnsemble
- over-sampling(过采样)
  - __SMOTE__(Synthetic Minority Over-sampling Technique)
  - Borderline-SMOTE，将少数类样本根据距离多数类样本的距离分为noise,safe,danger三类样本集，只对danger中的样本集合使用SMOTE算法
- 移动阈值

[How to handle Imbalanced Classification Problems in machine learning](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/)
[8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

---
<a id='experiences'></a>
# Experiences
- 自动化
- 封装性
- 保存所有log
- 保存模型 -> 复现
- seed
  - train_seed
  - cv_seed
  - global_seed
- 关于比赛
  - 泛化性能衡量
  - 规则、先验知识

---
<a id='materials'></a>
# Materials

## Books
- PRML(Pattern Recognition and Machine Learning)
- EoS(The Elements of Statistical Learning)
- CO(Convex Optimization)
- 统计学习方法（李航，清华大学出版社）
- 机器学习（周志华， 清华大学出版社）- 西瓜书
- 机器学习实战（Peter Harrington，人民邮电出版社出版）

## Links
- [UFLDL Tutorial: Unsupervised Feature Learning and Deep Learning](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial)
- [Best Machine Learning Resources for Getting Started](https://machinelearningmastery.com/best-machine-learning-resources-for-getting-started/)
- [A Tour of Machine Learning Algorithms](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
- [Machine Learning GitHub](https://github.com/JustFollowUs/Machine-Learning)
- [How to start doing Kaggle competitions？]( https://www.quora.com/How-do-I-start-participating-in-Kaggle-competitions-What-basic-data-analysis-do-people-do-and-how-do-people-know-which-models-to-apply-How-do-they-make-improvements)
- [What do top Kaggle competitors focus on?](https://www.quora.com/What-do-top-Kaggle-competitors-focus-on-What-helped-them-do-better-than-others)
- [A Journey Into Data Science](https://ajourneyintodatascience.quora.com/)
- [Techniques to improve the accuracy of your Predictive Models](http://anotherdataminingblog.blogspot.jp/2013/10/techniques-to-improve-accuracy-of-your_17.html)

##网盘：
- https://pan.baidu.com/s/1skU0xEt
- 密码：nxnc
