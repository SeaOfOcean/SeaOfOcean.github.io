---
layout: post
title: AutoML自动超参调优
tags:
  - AutoML
  - Hyper Parameter
---

现在AutoML非常的火，各大云平台都在推出自己的AutoML服务，包括Google Cloud，Amazon SageMaker，MS Azure等等。AutoML要解决的问题主要是释放机器学习过程中的人力投入，包括：
* 数据预处理
* 特征提取
* 模型选择/模型构建
*  **模型超参优化**
* 结果的评估

本文主要关注的是**模型超参优化**的自动化。

## 调参系统架构

AutoML自动调参又可以称作黑箱超参优化（Blackbox hyperparameter optimization）。比较常见的做法是将调参系统和训练系统分离开，模型、数据和训练过程由用户来控制，调参系统给训练系统建议一组或多组参数，训练系统反馈结果，然后调参系统根据反馈结果产生下一组建议的参数。这个过程一直迭代直至满足了终止条件。
![调参系统架构]({{ site.baseurl }}/images/hyper-opt.png)

## 调参算法
调参算法的输入是用户指定的参数及其范围，比如设定学习率范围为[0.0001, 0.01]。比较常见的算法为网格搜索，随机搜索和贝叶斯优化等。

1. Grid search

遍历所有可能的参数组合。网格搜索很容易理解和实现，例如我们的超参数A有2种选择，超参数B有3种选择，超参数C有5种选择，那么我们所有的超参数组合就有2 * 3 * 5也就是30种，我们需要遍历这30种组合并且找到其中最优的方案，对于连续值我们还需要等间距采样。实际上这30种组合不一定取得全局最优解，而且计算量很大很容易组合爆炸，并不是一种高效的参数调优方法。

2. Random search

限定搜索次数，随机选择参数进行实验。业界公认的Random search效果会比Grid search好，Random search其实就是随机搜索，例如前面的场景A有2种选择、B有3种、C有5种、连续值随机采样，那么每次分别在A、B、C中随机取值组合成新的超参数组合来训练。虽然有随机因素，但随机搜索可能出现效果特别差、也可能出现效果特别好，在尝试次数和Grid search相同的情况下一般最值会更大

3. Bayesian Optimization

业界的很多参数调优系统都是基于贝叶斯优化的，如Google Vizier [1], SigOpt[2].
该算法要求已经存在几个样本点（一开始可以采用随机搜索来确定几个初始点），并且通过高斯过程回归（假设超参数间符合联合高斯分布）计算前面n个点的后验概率分布，得到每一个超参数在每一个取值点的期望均值和方差，其中均值代表这个点最终的期望效果，均值越大表示模型最终指标越大，方差表示这个点的效果不确定性，方差越大表示这个点不确定是否可能取得最大值非常值得去探索。

## Early Stop算法

在调参的过程中，有的参数在训练的过程中，观察曲线的趋势，会发现它是不太有希望在训练结束达成目标的，这个时候，将这些任务终止掉，释放资源，继续别的参数的尝试。这样可以快速试错，快速调整。[1]

1. Performance Curve Stopping Rule: 对训练曲线做回归分析，对于一段曲线，预测最终能达到的准确率，如果太低，则停掉。

2. Median Stopping Rule: 在Step s的准确率如果小于其他观测到的训练曲线在step s处的平均值，则将这次尝试停掉。


## 开源调参框架
我调研了Github上开源的超参调优系统，按照受关注程度排序如下：



| 调参框架 | GitHub | star | 算法 | 语言 | 

|:-----|:----|:----:|:----|:-----:|
|  Hyperopt   |   https://github.com/hyperopt/hyperopt |  2358   |  1. Random Search  2. Tree of Parzen Estimators (TPE)  3. 设计中，未实现：  Bayesian optimization   | Python |  
| BayesianOptimization  | https://github.com/fmfn/BayesianOptimization  | 1687  | Bayesian Optimization | Python  
| Spearmint|https://github.com/HIPS/Spearmint| 1062| Bayesian Optimization  | Python
|Advisor |  https://github.com/tobegit3hub/advisor  |  320  | 1. Random Search Algorithm  2. Grid Search Algorithm  Baysian Optimization  3. Early Stop  | Python  
| RoBO  | https://github.com/automl/RoBO  |230  |  Bayesian Optimization   | Python |
| SMAC3   |  https://github.com/automl/SMAC3   | 208   | Baysian Optimization   | Python3  |
|  Bayesopt   |  https://github.com/rmcantin/bayesopt   |  163   |   Bayesian Optimization  |  Python  |
|  Autoweka |  https://github.com/automl/autoweka   |   146   |   Baysian Optimization   | Java   |

## 总结
现在的调参系统基本上都是基于贝叶斯优化的思想，将调参任务作为一个黑箱优化的问题。

## Reference
1. [Google Vizier A Service for Black-Box Optimization](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/bcb15507f4b52991a0783013df4222240e942381.pdf)
2. [SigOpt: https://sigopt.com/](https://sigopt.com/)
