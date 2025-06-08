# Data-Analysis-with-Python

## Reference

- [Student Sleep Patterns - Kaggle](https://www.kaggle.com/code/orhansere/student-sleep-patterns-kmeans-87-score)

## 预处理

1. 调用适当的库

<div align="center">

| 类别         | 库名称             |
|--------------|-------------------|
| 数据分析库   | **_pandas_**      |
|              | matplotlib        |
| 可视化作图库 | pyecharts         |
|              | seaborn           |
| sklearn库    | **_sklearn_**     |
| 其他库       | warning           |

</div>

3. 读取.CSV文件
4. 去除多余信息
5. 提取关键信息

## 初步结果分析

1. 结果分析(Sleep Quality with **Columns**)
2. 结果分析(Which **Gender** take more caffeine?)
3. 结果分析(**Disturbution of University Year** \& **Gender**)

## 模型训练

1. 对"dataframe"中的数据进行"**独热编码**"
2. 特征标准化
3. 归一化操作
4. 利用PCA(主成分分析)对高维特征进行降维处理

## 最终结果分析

1. 结果分析()
2. 通过计算轮廓系数(Silhouette Score)确定 K-Means 聚类的最佳簇数(K值)
3. 通过计算轮廓系数(Silhouette Score)评估不同簇数(K值)下的K-means聚类效果
4. 使用 K-Means 对数据进行聚类分析
5. 输出 K-Means 聚类模型的SSE(Sum of Squared Errors, 误差平方和)以评估聚类效果
6. 计算并输出 K-Means 聚类效果的轮廓系数(Silhouette Score)以评估聚类效果的质量
7. 结果分析()
8. 对聚类结果进行分析和解释
