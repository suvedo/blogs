[*<<返回主页*](../index.md)<br><br>
**本文为作者原创，转载请注明出处**<br>
本文结合作者对xgboost原理的理解及使用xgboost做分类问题的经验，讲解xgboost在分类问题中的应用。内容主要包括xgboost原理简述、xgboost_classifer代码、xgboost使用心得和几个深入的问题<br>
### xgboost原理简述
xgboost基本思想是叠加多个基分类器的结果组合成一个强分类器，叠加方法就是各个分类器的结果做加法，在生成下一个基分类器的时候，目标是拟合历史分类器结果之和与label之间的残差，预测的时候是将每个基分类器结果相加；每个基分类器都是弱分类器，目前xgboost主要支持的基分类器有CART回归树、线性分类器；<br><br>
#### CART回归树
CART(Classification And Regression Tree)回归树，顾名思义，是一个回归模型，同[C4.5](https://zh.wikipedia.org/wiki/C4.5%E7%AE%97%E6%B3%95)分类树一样，均是二叉树模型；父节点包含所有的样本，每次分裂的时候将父节点的样本划分到左子树和右子树中，划分的原则是找到最优的特征的最优划分点使得目标函数最小，CART回归树的目标函数是平方误差，而C4.5的目标函数是信息增益（划分后的信息增益最大）；
对CART回归树来说，孩子节点中样本的预测值是所有样本预测值的平均值（why？因为CART的目标函数是平方误差，是的平方误差最小的预测值就是平均值，可以证明一下），
而C4.5决策树中孩子节点的预测值一般采用投票法；在构建树的时候，两者都采用贪心算法，即每次节点分裂时找本次所有分裂点中目标函数最小的分裂点，该方法不一定能找到全局最优的树结构，但能有效的降低计算量<br>
#### xgboost
多数情况下，将CART回归树作为xgboost的基分类器（tree-based xgbooster），xgboost不断生成新的CART回归树，每生成一颗树即是在学习一个新函数，这个函数将每个样本映射到唯一确定的一个叶子节点中，所有叶子节点中的样本共享相同的预测值，函数的目标则是去拟合样本的残差，损失函数可以是与CART回归树相同的均方误差，也可以是交叉熵（一般用于分类问题中）、pairwise loss（一般用于rank问题中）；<br><br>
xgboost的目标函数可以表示如下：<br>
![xgb_obj_1](../images/NLP/6_xgboost_classifier/xgb_obj_1.png)<br>
其中第一项是训练损失（平方误差、交叉熵等），第二项是正则化损失（L1、L2等）；为了便于计算，对上式进行泰勒展开，并取0/1/2阶项作为目标函数的近似表示：<br>
![xgb_obj_2](../images/NLP/6_xgboost_classifier/xgb_obj_2.png)<br>
将正则化项（L1和L2正则化项均不为0，其系数分别为gamma和lambda）带入上式，并进一步化简（将各个叶子节点中样本综合）得到如下：<br>
![xgb_obj_3](../images/NLP/6_xgboost_classifier/xgb_obj_3.png)<br>
不难发现，这个函数是关于叶子节点权重w<sub>j</sub>的二次函数，其最值和最值点分别为：<br>
![xgb_obj_4](../images/NLP/6_xgboost_classifier/xgb_obj_4.png)<br>
这样近似及化简之后，针对每个候选划分能快速的计算其孩子节点预测值和目标函数值；<br><br>
更详细的推导请参考[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754v1.pdf)<br><br>和陈天奇的演讲ppt;<br><br>
### 代码使用方法
代码地址：https://github.com/suvedo/xgboost_classifier
#### 线下训练
首先按照[xgboost官方教程](https://xgboost.readthedocs.io/en/latest/build.html)clone源码，并安装python包。（不详述）；<br><br>
配置train_dir/config.py中的参数，包括：1）XGBoostConifg中的与xgboost模型相关的参数（参数含义详见[XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)，2）Config中的与输入输出及其它训练相关的参数；<br><br>
配置完参数运行sh run.sh即可训练、验证并测试模型，程序会保存训练好的模型，将训练好的模型拷贝到deploy/c++目录下即可在线预测<br><br>
#### 在线预测
deploy/c++目录下运行make编译代码，然后运行./test_xgb_cls即可预测，预测之前先配置conf/xgb_cls.conf，包括模型路径、特征维度、树数量；若要把在线预测代码集成在自己的代码中，只需要拷贝 include/, lib/, xgboost_classifier.h, xgboost_classifier.cpp即可<br><br>
### 使用心得
#### 调参心得
booster：指明使用的基分类器，默认为gbtree，还可以选择gblinear和dart。gbtree则表示CART树，gblinear表示线性分类器，dart也是使用CART作为基分类器，只不过对各个CART树使用类似与nn里的drop-out，可以配置rate_drop来指明drop-out rate；一般情况下，都使用默认的gbtree;<br><br>
eta：收缩因子，或者学习率，每颗树的结果乘以eta为样本在这颗树的最终得分，eta一般小于1，可以给后续的基分类器更大的学习空间，也可以避免过拟合，如果发现树的颗树比较少，可以适当调低eta；默认值为0.3；<br><br>
gamma：参数中的gamma不是公式中的L1正则化系数（L1正则化系数为alpha），而是最小的分类损失降低，只有当节点分裂带来的损失大于gamma时才进行分裂，可以有效避免过拟合；默认值为0，注意：因为有正则化项的存在，分裂节点不一定能带来正向的损失减小，所以gamma为0不一定表示所有的分裂均满足要求（均能分裂）；<br><br>
max_depth：树的最大深度，这个比较好理解；默认值为6，如果过拟合严重，可以适当减小该参数值；<br><br>
subsample：生成下一颗树时训练样本的采样率，类似随机森林，默认为1，如果样本数量比较大或者过拟合严重，可以考虑增大该参数值；<br><br>
colsample_bytree：生成每一颗树时对特征的采样率，类似随机森林，默认为1<br><br>
colsample_bylevel：生成每一层时特征的采用率，在每颗树的特征的基础上采样，colsample_bytree*colsample_bylevel，默认值为1<br><br>
colsample_bynode：节点分裂时特征的采样率，在每颗树、每层的基础上采样，colsample_bytree*colsample_bylevel*colsample_bynode，默认值为1<br><br>
lambda：叶子节点输出值的L2正则化系数，默认为1<br><br>
alpha：叶子节点输出值的L1正则化系数，默认为0，即不做L1正则化系数<br><br>
objective：目标函数，默认为reg:squarederror，可以取binary:logistic/multi:softmax/rank:pairwise等；我在最先调参的时候使用reg:squarederror，而线上的xgboost版本比较老旧，不支持此目标函数，因为只能换成binary:logistic重新训练，换成binary:logistic并且early-stop的eval_metric选用auc反倒效果变好，分析原因为：我的任务中正负例样本比例悬殊比较大，使用rmse、error等对正负例敏感的eval_metric反倒效果好，auc的含义及计算方法见[机器学习一般流程总结](../NLP/3_ml_process.md)；<br><br>
eval_metric：验证集的metric，在训练的日志中能看到train和eval的metric；根据目标函数设置默认值；<br><br>
enable_early_stop：是否使用early-stop，一般都需要使用验证集做训练时的验证和早停，避免过拟合，也可以通过早停的情况了解自己的模型是否过拟合了；如果在xgb.train()的参数中enable了早停，则必须要指定evals列表；<br><br>
early_stopping_rounds：eval_metric在early_stopping_rounds轮没有增加或减少则停止训练，一般设置为10轮<br><br>
#### xgboost有哪些优点
我最先使用nn做分类问题，nn比较擅长特征抽取，而我的任务的特征基本上是抽取好的，不需要nn的特征抽取能力，且nn训练慢，必须依赖gpu（公司内部训练工具导致），任务排队比较耗时；后来选用xgboost，训练速度极快，且不依赖gpu，效果与nn基本持平；另外还有一点xgboost可以计算每个特征的重要性（get_fscore()或get_score()接口），这对于特征筛选、模型可解释性、模型透明、模型调优等都有好处；计算特征的重要性的原理：通过指明的important_type计算，比如important_type=weight，则重要性就是该特征被用于节点分裂的次数，important_type=gain则表示特征分裂带来的平均收益；xgboost还可以以明文的形式保存树模型，方便模型可视化和调优；<br><br>
### 几个有深度的问题
问题一：特征分裂点怎么找的？类别特征怎么处理？<br><br>
问题二：孩子节点的值计算都是平均值吗？还是针对不同的loss有不同的计算方法？<br><br>
问题三：xgboost如何并行？<br><br>
### 参考文献
[xgboost官方教程](https://xgboost.readthedocs.io/en/latest/index.html)<br><br>
[一文读懂机器学习大杀器XGBoost原理](https://zhuanlan.zhihu.com/p/40129825)<br><br>
[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754v1.pdf)<br><br>
