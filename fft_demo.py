import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # sr = 8000
    # l = 3
    # t = sr * l
    # t = np.array(range(0,t)) / sr
    # x = np.sin(2 * np.pi * 130 * t) + 0.8 * np.sin(2 * np.pi* 160 * t) + 1.2 * np.sin(2 * np.pi * 190 * t)
    # x = x/ max(x)
    #
    # y = np.fft.fft(x)
    #
    # fr = np.array(range(0, len(x)))
    # region = int(len(fr)/2)
    # # plt.plot(fr[0:region], abs(y[0:region]))
    # plt.plot(fr, abs(y))
    # plt.show()

    import pandas as pd
    import seaborn as sns
    from scipy.cluster import hierarchy

    # 创建一个10行5列的随机数据DataFrame
    np.random.seed(0)
    data = np.random.randn(100, 5)
    df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])

    # 计算DataFrame的相关系数矩阵
    corr = df.corr()

    # 计算距离矩阵和聚类树
    dist = 1 - corr.abs()  # 使用距离矩阵进行聚类
    linkage = hierarchy.linkage(dist, method='ward')

    # 对相关性矩阵进行重新排序
    order = hierarchy.leaves_list(linkage)
    corr = corr.iloc[order, order]

    # 绘制相关性热图
    sns.clustermap(corr, cmap='rainbow', row_cluster=True, col_cluster=True, annot=True)

    # 显示图形
    plt.show()

    # import seaborn as sns
    # import numpy as np
    # import pandas as pd
    # from scipy.spatial.distance import pdist, squareform
    # from scipy.cluster import hierarchy
    #
    # # 构造一个随机数据矩阵
    # np.random.seed(0)
    # data = np.random.randn(10, 5)
    #
    #
    # df = pd.DataFrame(data, columns=["a",'b','c','d','e'])
    # corr = df.corr()
    # sns.clustermap(corr, method="ward", metric="euclidean", figsize=(8,8))
    # linkage=hierarchy.ward(corr)
    # dendo = hierarchy.dendrogram(linkage, leaf_rotation=90)
    # plt.show()

    # # 计算数据的皮尔逊相关系数
    # corr = np.corrcoef(data, rowvar=False)
    #
    # # 计算数据的欧几里得距离
    # dist = pdist(data)
    #
    # # 对距离进行层次聚类
    # linkage_matrix = linkage(dist, method='ward')
    #
    # # 根据聚类结果重新排序矩阵
    # # clusters = fcluster(linkage_matrix, 3, criterion='maxclust')
    # # idx = np.argsort(clusters)
    # # corr = corr[idx, :][:, idx]
    #
    # # 绘制带聚类结果的相关系数热图
    # sns.clustermap(corr, row_cluster=True, col_cluster=True, annot=True)
    # plt.show()
