import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
from datetime import datetime

# class AppAnalytics:
#     def __init__(self, data_path):
#         self.data = pd.read_excel(data_path)  # 从CSV文件中加载数据
#         self.data['数据日期'] = pd.to_datetime(self.data['数据日期'])  # 将时间列转换为日期时间格式
#
#     def calculate_metrics(self):
#         # 计算常用运营指标
#         metrics = {}
#
#         # 总访问次数
#         metrics['总访问次数'] = len(self.data)
#
#         # 按天统计总访问次数
#         daily_visits = self.data.groupby(self.data['数据日期'].dt.strftime('%Y-%m-%d'))['二级机构名称'].count()
#         metrics['按天统计总访问次数'] = daily_visits.to_dict()
#
#         # 不同机构的访问次数
#         org_visits = self.data['二级机构名称'].value_counts().to_dict()
#         metrics['不同机构的访问次数'] = org_visits
#
#         # 不同页面的访问次数
#         page_visits = self.data['一级页面名称'].value_counts().to_dict()
#         metrics['不同页面的访问次数'] = page_visits
#
#         org_daily_visits = self.data.groupby(['二级机构名称', self.data['数据日期'].dt.strftime('%Y-%m-%d')])[
#             '二级机构名称'].count().unstack()
#         metrics['不同机构不同天的访问次数'] = org_daily_visits#.fillna(0, inplace=True)
#
#         org_daily_visits = self.data.groupby(['一级页面名称', self.data['数据日期'].dt.strftime('%Y-%m-%d')])[
#             '一级页面名称'].count().unstack()
#         metrics['不同一级页面名称不同天的访问次数'] = org_daily_visits
#         return metrics
#
#     def plot_metrics(self):
#         metrics = self.calculate_metrics()
#
#         # 绘制按天统计总访问次数和不同机构的访问次数
#         daily_visits = pd.Series(metrics['按天统计总访问次数'])
#         org_visits = pd.Series(metrics['不同机构的访问次数'])
#
#         fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
#         daily_visits.plot(ax=axes[0], marker='o')
#         axes[0].set_title('按天统计总访问次数')
#         axes[0].set_xlabel('日期')
#         axes[0].set_ylabel('访问次数')
#
#         org_visits.plot(kind='bar', ax=axes[1])
#         axes[1].set_title('不同机构的访问次数')
#         axes[1].set_xlabel('机构')
#         axes[1].set_ylabel('访问次数')
#
#         plt.tight_layout()
#         plt.show()
#
#     def plot_top_10_orgs(self):
#         metrics = self.calculate_metrics()
#
#         # 获取总访问次数前十的机构
#         top_10_orgs = self.data['二级机构名称'].value_counts().head(6).index.tolist()
#
#         # 绘制总访问次数前十的机构在按天统计的访问次数折线图
#         org_daily_visits = metrics['不同机构不同天的访问次数']
#         org_daily_visits.fillna(0, inplace=True)
#         fig, ax = plt.subplots(figsize=(12, 6))
#         for org in top_10_orgs:
#             # if org in org_daily_visits:
#             org_daily_visits.loc[org].plot(ax=ax, marker='o', label=org)
#
#         ax.set_title('总访问次数前十的机构在按天统计的访问次数')
#         ax.set_xlabel('日期')
#         ax.set_ylabel('访问次数')
#         ax.legend()
#         ax.xaxis.set_major_locator(plt.MaxNLocator(12))  # 设置横轴日期刻度的间隔
#         plt.xticks(rotation=45)  # 旋转日期标签以避免重叠
#         plt.tight_layout()
#         plt.show()
#
#     def plot_yiji_orgs(self):
#         metrics = self.calculate_metrics()
#
#
#
#         # 绘制总访问次数前十的机构在按天统计的访问次数折线图
#         org_daily_visits = metrics['不同一级页面名称不同天的访问次数']
#         org_daily_visits.fillna(0, inplace=True)
#         fig, ax = plt.subplots(figsize=(12, 6))
#         for org in set(self.data["一级页面名称"]):
#             # if org in org_daily_visits:
#             org_daily_visits.loc[org].plot(ax=ax, marker='o', label=org)
#
#         ax.set_title('总访问次数前十的机构在按天统计的访问次数')
#         ax.set_xlabel('日期')
#         ax.set_ylabel('访问次数')
#         ax.legend()
#         ax.xaxis.set_major_locator(plt.MaxNLocator(12))  # 设置横轴日期刻度的间隔
#         plt.xticks(rotation=45)  # 旋转日期标签以避免重叠
#         plt.tight_layout()
#         plt.show()
# if __name__ == "__main__":
#     data_path = 'your_data.csv'  # 替换为你的数据文件路径
#     app_analytics = AppAnalytics(data_path)
#     app_analytics.plot_metrics()



import pandas as pd

import seaborn as sns

# class AppAnalytics:
#     def __init__(self, data_path):
#         self.data = pd.read_excel(data_path)  # 从Excel文件中加载数据
#         self.data['数据日期'] = pd.to_datetime(self.data['数据日期'])  # 将时间列转换为日期时间格式
#
#     def calculate_metrics(self):
#         # 计算每一天不同二级机构名称和一级页面名称的访问数
#         daily_metrics = self.data.groupby(['数据日期', '二级机构名称', '一级页面名称']).size().unstack(fill_value=0)
#
#         # 计算不同月份每一天不同二级机构名称的访问数，按访问数降序排序
#         org_metrics_by_month = {}
#         for month in range(7, 10):
#             tmp = []
#             for i, ttt in daily_metrics.iterrows():
#                 if i[0].month == month:
#                     tmp.append(True)
#                 else:
#                     tmp.append(False)
#             data_month = daily_metrics[tmp]
#             data_tmp = data_month.groupby('二级机构名称').sum()
#             org_metrics_by_month[month] = data_tmp
#             # org_metrics = data_month.groupby('二级机构名称').sum().sum(axis=1).sort_values(ascending=False)
#             # org_metrics_by_month[month] = data_month[org_metrics.index]
#
#         return org_metrics_by_month
#
#     def plot_metrics(self, org_metrics_by_month):
#         # 使用seaborn设置样式
#         sns.set(style="whitegrid",font_scale=1.5,font='STSong')
#
#         # 绘制每个月的柱状图，不同颜色的段表示不同一级页面名称的访问数
#         for month, org_metrics in org_metrics_by_month.items():
#
#             org_metrics["访问总数"] = org_metrics["交易规模"] + org_metrics["实时监控"]+ org_metrics["客户规模"]
#             org_metrics.sort_values(by="访问总数", ascending=False, inplace=True)
#             plt.figure(dpi=200, figsize=(50, 8))
#             org_metrics[["交易规模", "实时监控", "客户规模"]].plot(kind='bar', stacked=True, rot=45, cmap='tab20', width=0.8)
#
#             plt.title(f'不同二级机构名称的访问数 ({month}月)')
#             plt.xticks(fontsize=5)
#             plt.xlabel('二级机构名称')
#             plt.ylabel('访问数')
#             # plt.legend(title='一级页面名称', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
#
#             plt.tight_layout()
#             plt.savefig("./{}.jpg".format(month), dpi=330)
#             # plt.show()

class AppAnalytics:
    def __init__(self, data_path):
        self.data = pd.read_excel(data_path)  # 从Excel文件中加载数据
        self.data['数据日期'] = pd.to_datetime(self.data['数据日期'])  # 将时间列转换为日期时间格式

    def calculate_metrics(self):
        # 计算每一天不同二级机构名称和一级页面名称的访问数
        daily_metrics = self.data.groupby(['数据日期', '二级机构名称', '二级页面名称']).size().unstack(fill_value=0)

        # 计算不同月份每一天不同二级机构名称的访问数，按访问数降序排序
        org_metrics_by_month = {}
        for month in range(7, 10):
            tmp = []
            for i, ttt in daily_metrics.iterrows():
                if i[0].month == month:
                    tmp.append(True)
                else:
                    tmp.append(False)
            data_month = daily_metrics[tmp]
            data_tmp = data_month.groupby('二级机构名称').sum()
            org_metrics_by_month[month] = data_tmp
            # org_metrics = data_month.groupby('二级机构名称').sum().sum(axis=1).sort_values(ascending=False)
            # org_metrics_by_month[month] = data_month[org_metrics.index]

        return org_metrics_by_month

    def plot_metrics(self, org_metrics_by_month):
        # 使用seaborn设置样式
        sns.set(style="whitegrid",font_scale=1.5,font='STSong')

        # 绘制每个月的柱状图，不同颜色的段表示不同一级页面名称的访问数
        for month, org_metrics in org_metrics_by_month.items():

            org_metrics["访问总数"] = org_metrics["套餐-A"] + org_metrics["套餐-B"]+ org_metrics["套餐-C"]
            org_metrics.sort_values(by="访问总数", ascending=False, inplace=True)
            plt.figure(dpi=200, figsize=(50, 8))
            org_metrics[["套餐-A", "套餐-B", "套餐-C"]].plot(kind='bar', stacked=True, rot=45, cmap='tab20', width=0.8)

            plt.title(f'不同二级机构名称的访问数 ({month}月)')
            plt.xticks(fontsize=5)
            plt.xlabel('二级机构名称')
            plt.ylabel('访问数')
            # plt.legend(title='一级页面名称', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)

            plt.tight_layout()
            plt.savefig("./{}-1.jpg".format(month), dpi=330)
            # plt.show()

if __name__ == "__main__":
    data_path = r'C:\ShareCache\李燚航_2001111828\2023秋\工作\中信作业.xlsx' # 替换为你的Excel文件路径
    app_analytics = AppAnalytics(data_path)
    org_metrics_by_month = app_analytics.calculate_metrics()

    # 绘制每个月的柱状图，不同颜色的段表示不同一级页面名称的访问数
    app_analytics.plot_metrics(org_metrics_by_month)



# if __name__ == "__main__":
#     data_path = r'C:\ShareCache\李燚航_2001111828\2023秋\工作\中信作业.xlsx'  # 替换为你的数据文件路径
#     app_analytics = AppAnalytics(data_path)
#     # app_analytics.plot_top_10_orgs()
#     data_by_month = app_analytics.calculate_metrics()
#
#     # 绘制每个月的访问数折线图
#     app_analytics.plot_metrics(data_by_month)
#     # app_analytics.plot_yiji_orgs()
