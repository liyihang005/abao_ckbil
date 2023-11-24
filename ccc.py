from typing import List


# class Solution:
#     def countSquares(self, matrix: List[List[int]]) -> int:
#         dp_table=[[0] * len(matrix[0])] * len(matrix)
#         dp_table[0] = matrix[0]
#         for i in range(len(dp_table)):
#             print(matrix[i][0])
#             dp_table[i][0] = matrix[i][0]
#             print('******')
#             print(dp_table[i][0])
#
#         res = 0
#         for j in range(1, len(matrix[0])):
#             for i in range(1, len(matrix)):
#                 if matrix[i][j] == 0:
#                     dp_table[i][j] = 0
#                 else:
#                     dp_table[i][j] = min(dp_table[i-1][j], dp_table[i][j-1], dp_table[i-1][j-1]) + 1
#                 res += dp_table[i][j]
#         return res + sum(matrix[0]) + sum([matrix[ii][0] for ii in range(len(matrix))]) - matrix[0][0]

# class Solution:
#     def permute(self, nums: List[int]) -> List[List[int]]:
#         self.res = []
#         self.path = []
#         def backstracking(nums,used):
#             if len(self.path) == len(nums):
#                 self.res.append(self.path[:])
#                 return None
#             for index, element in enumerate(used):
#                 if used[index]:
#                     continue
#                 self.path.append(nums[index])
#                 used[index] = True
#                 backstracking(nums, used)
#                 self.path.pop(-1)
#                 used[index] = False
#             return None
#
#         backstracking(nums, [False]*len(nums))
#         return self.res

# class Solution:
#     def islandPerimeter(self, grid: List[List[int]]) -> int:
#         perimeter = 0
#         # 在地图周围再加上一圈“水”
#         new_map = [[0] * (len(grid[0]) + 2)] * (len(grid) + 2)
#         for i in range(1, len(new_map) - 1):
#             new_map[i] = [0] + grid[i-1] + [0]
#
#         for i in range(1, len(grid)+1):
#             for j in range(1, len(grid[0])+1):
#                 if new_map[i][j] > 0:
#                     perimeter += (4 - sum([new_map[i-1][j], new_map[i+1][j], new_map[i][j-1], new_map[i][j+1]]))
#         return perimeter

# class Solution:
#     def findComplement(self, num: int) -> int:
#         from math import ceil, sqrt, log
#         # print((num >> 1) << 2)
#         # return ((num >> 1) << 2) - 1- num
#         return 2 ** (ceil(log(num, 2))) - 1 - num if 2 ** (ceil(log(num, 2))) - 1 - num >= 0 else 2 ** (ceil(log(num, 2) + 0.000001)) - 1 - num


# class Solution:
# #     def reachNumber(self, target: int) -> int:
# #         if target < 0:
# #             target *= -1
# #         tmp_sum = 0
# #         for i in range(int(1e9)):
# #             if tmp_sum < target:
# #                 tmp_sum += (i + 1)
# #             elif tmp_sum == target:
# #                 return i
# #             elif (tmp_sum - target) % 2 == 0:
# #                 return i
# #             else:
# #                 tmp_sum += (i + 1)

# class Solution:
#     def massage(self, nums: List[int]) -> int:
#
#         dp = [nums[0]] * len(nums)
#         if nums[1] >= nums[0]:
#             dp[1] = nums[1]
#         for i in range(2, len(nums)):
#             dp[i] = max([tmp + nums[i] for tmp in dp[:i-1]])
#         return max(dp)


# class Solution:
#     def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
#         matrix = []
#         for i in range(n):
#             tmp = []
#             for j in range(n):
#                 tmp.append(1)
#             matrix.append(tmp)
#
#         for pos in mines:
#             matrix[pos[0]][pos[1]] = 0
#
#
#         dp_table = matrix
#         def numbers_of_1(mat, i, j):
#             left = 0
#             for tmp_i in range(i-1, -1, -1):
#                 if mat[tmp_i][j] > 0:
#                     left += 1
#                 else:
#                     break
#             right = 0
#             for tmp_i in range(i + 1, len(mat), 1):
#                 if mat[tmp_i][j] > 0:
#                     right += 1
#                 else:
#                     break
#             up = 0
#             for tmp_i in range(j + 1, len(mat), 1):
#                 if mat[i][tmp_i] > 0:
#                     up += 1
#                 else:
#                     break
#             down = 0
#             for tmp_i in range(j - 1, -1, -1):
#                 if mat[i][tmp_i] > 0:
#                     down += 1
#                 else:
#                     break
#             return min(up, down, left, right)
#         print(numbers_of_1(matrix, 1, 1))
#         for i in range(1, n - 1):
#             for j in range(1, n - 1):
#                 if matrix[i][j] > 0:
#                     dp_table[i][j] = numbers_of_1(matrix, i, j) + 1
#
#         all_num = []
#         for r in dp_table:
#             for c in r:
#                 all_num.append(c)
#         return max(all_num)

# class Solution:
#     def convertInteger(self, A: int, B: int) -> int:
#         if A < 0:
#             A = (1 << 32) + A
#         if B < 0:
#             B = (1 << 32) + B
#             # 去掉前缀ob
#
#         strA = format(A, "032b")
#         strB = format(B, "032b")
#         num = 0
#         for i in range(32):
#             if strA[i] != strB[i]:
#                 num = num + 1
#         return num

# class Solution:
#     def numTilings(self, n: int) -> int:
#         #看最后一列是怎么摆放的
#         #****@ |  **@ @@ |  **@@ @ | ***@@ | **@  @@ @ | **@@  @@
#         #****@ |  **@@ @ |  **@ @@ | ***@@ | **@@   @@ | **@ @@ @
#
#         if n == 1:
#             return 1
#         elif n == 2:
#             return 2
#         elif n == 3:
#             return 5
#         elif n == 4:
#             return 11
#         elif n == 0:
#             return 1
#
#         return self.numTilings(n-1) + self.numTilings(n-3) *2 + self.numTilings(n-2) + self.numTilings(n-4) * 2


# class Solution:
#     def isIdealPermutation(self, nums: List[int]) -> bool:
#         sorted_nums = sorted(nums)
#         for i in range(len(nums)):
#             if abs(nums[i] - sorted_nums[i]) > 1:
#                 return False
#         return True
# class Solution:
#     def compressString(self, S: str) -> str:
#         if len(S) < 1:
#             return S
#         tmp = S[0]
#         tmp_n = 1
#         i = 1
#         res = ''
#         while i < len(S):
#             if S[i] == tmp:
#                 tmp_n += 1
#             else:
#                 res += tmp + str(tmp_n)
#                 tmp = S[i]
#                 tmp_n = 1
#             i += 1
#         res += tmp + str(tmp_n)
#         return res
# import numpy as np
# class Solution:
#     def minOperations(self, boxes: str) -> List[int]:
#         dp_table = []
#         for i in range(len(boxes)):
#             tmp = []
#             for j in range(len(boxes)):
#                 if j == i+1:
#                     tmp.append(int(boxes[i]))
#                 elif j == i-1:
#                     tmp.append(int(boxes[i]))
#                 else:
#                     tmp.append(0)
#             dp_table.append(tmp)
#
#
#         for i in range(len(boxes)):
#             for j in range(i+2, len(boxes)):
#                 dp_table[i][j] = dp_table[i][j-1] + int(boxes[i])
#
#         for i in range(len(boxes)):
#             for j in range(i-2, -1, -1):
#                 dp_table[i][j] = dp_table[i][j+1] + int(boxes[i])
#
#         res = []
#         for j in range(len(boxes)):
#             tmp = []
#             for i in range(len(boxes)):
#                 tmp.append(dp_table[i][j])
#             res.append(sum(tmp))
#         return res


# class Solution:
#
#     def readBinaryWatch(self, turnedOn: int) -> List[str]:
#         top = [8, 4, 2, 1]
#         bottom = [32, 16, 8, 4, 2, 1]
#         num = top + bottom
#         self.res = []
#         self.path = []
#
#         def backstracking(nums, used):
#             if len(self.path) == turnedOn:
#                 self.res.append(self.path[:])
#                 return
#             for i, ele in enumerate(nums):
#                 if used[i]:
#                     continue
#                 else:
#                     self.path.append(nums[i])
#                     used[i] = True
#                     backstracking(nums, used)
#                     self.path.pop(-1)
#                     used[i] = False
#
#         def parse_res(res):
#             top = [8, 4, 2, 1]
#             bottom = [32, 16, 8, 4, 2, 1]
#             num = top + bottom
#             ret = []
#             for r in res:
#                 h = sum([num[tmp] for tmp in r if tmp < 4])
#                 m = sum([num[tmp] for tmp in r if tmp >= 4])
#                 str_m = str(m)
#                 if m <= 9:
#                     str_m = '0' + str(m)
#                 ret.append(str(h) + ':' + str_m)
#             return ret
#
#         backstracking([idx for idx in range(len(num))], [False] * len(num))
#         return parse_res(self.res)
#
# def aaaa():
#     # a = ["0:07","0:11","0:13","0:14","0:19","0:21","0:22","0:25","0:26","0:28","0:35","0:37","0:38","0:41","0:42","0:44","0:49","0:50","0:52","0:56","1:03","1:05","1:06","1:09","1:10","1:12","1:17","1:18","1:20","1:24","1:33","1:34","1:36","1:40","1:48","2:03","2:05","2:06","2:09","2:10","2:12","2:17","2:18","2:20","2:24","2:33","2:34","2:36","2:40","2:48","3:01","3:02","3:04","3:08","3:16","3:32","4:03","4:05","4:06","4:09","4:10","4:12","4:17","4:18","4:20","4:24","4:33","4:34","4:36","4:40","4:48","5:01","5:02","5:04","5:08","5:16","5:32","6:01","6:02","6:04","6:08","6:16","6:32","7:00","8:03","8:05","8:06","8:09","8:10","8:12","8:17","8:18","8:20","8:24","8:33","8:34","8:36","8:40","8:48","9:01","9:02","9:04","9:08","9:16","9:32","10:01","10:02","10:04","10:08","10:16","10:32","11:00"]
#
#     b = ["2:12","1:03","4:34","0:44","0:38","4:36","10:08","5:01","4:48","8:18","6:02","5:08","8:40","1:17","1:05","0:07","0:11","0:35","8:34","5:32","4:18","1:20","10:16","8:20","8:24","4:17","11:00","8:17","10:04","7:00","3:02","2:03","2:33","0:50","10:01","3:01","3:04","0:37","1:33","4:40","1:18","8:36","9:04","8:48","6:08","6:01","9:02","2:48","0:49","8:05","2:36","1:24","0:42","2:20","0:41","4:06","5:16","4:05","2:10","2:40","4:33","8:06","13:00","3:08","2:18","0:21","8:10","1:34","1:06","1:36","6:16","1:12","0:26","3:16","1:40","1:48","2:17","0:56","0:13","6:32","14:00","5:04","0:52","4:24","2:05","9:32","4:20","4:12","0:25","4:03","0:22","10:02","6:04","5:02","0:19","4:09","2:24","8:09","2:34","9:08","0:28","9:16","3:32","8:12","4:10","8:33","1:09","1:10","2:09","0:14","9:01","10:32","2:06","8:03"]
#     for i in b:
#         if i not in a:
#             print(i)


# class Solution:
#     def minimumLength(self, s: str) -> int:
#         i = 0
#         j = len(s) - 1
#
#         while i < j:
#             if s[i] != s[j]:
#                 return j - i + 1
#             tmp = s[i]
#             while j >= 0 and tmp == s[j]:
#                 j -= 1
#             while i <= len(s) - 1 and tmp == s[i]:
#                 i += 1
#         if i > j:
#             return 0
#         return j - i + 1

# class Solution:
#     def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
#         def ProjOverlap(line1: List[int], line2: List[int]) -> bool:
#             if line2[0] < line1[0]:
#                 line1, line2 = line2, line1
#             if (line2[0] >= line1[0] and line2[0] < line1[1]) or (line2[1] > line1[0] and line2[1] <= line1[1]):
#                 return True
#             return False
#         print(ProjOverlap([rec1[0], rec1[2]], [rec2[0], rec2[2]]))
#         print(ProjOverlap([rec1[1], rec1[3]],[rec2[1], rec2[3]]))
#         return ProjOverlap([rec1[0], rec1[2]], [rec2[0], rec2[2]]) and ProjOverlap([rec1[1], rec1[3]],
#                                                                                    [rec2[1], rec2[3]])

# class Solution:
# #     def lemonadeChange(self, bills: List[int]) -> bool:
# #         if bills[0] > 5:
# #             return False
# #         i = 0
# #         money = {'5':0, '10':0, '20':0}
# #         while i < len(bills):
# #             money[str(bills[i])] += 1
# #             if bills[i] == 5:
# #                 i += 1
# #                 continue
# #             elif bills[i] == 10:
# #                 if money['5'] < 1:
# #                     return False
# #                 else:
# #                     money['5'] -= 1
# #             elif bills[i] == 20:
# #
# #
# #                 if (money['10'] >= 1 and money['5'] >= 1):
# #                     money['5'] -= 1
# #                     money['10'] -= 1
# #                 elif money['5'] >= 3:
# #                     money['5'] -= 3
# #                 else:
# #                     return False
# #             i += 1
# #         return True


# class Solution:
#     def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
#         sum_alice = sum(aliceSizes)
#         sum_bob = sum(bobSizes)
#         alice_dict = {}
#         bob_dict = {}
#         for a in aliceSizes:
#             if not alice_dict.get(a):
#                 alice_dict[a] = 1
#         for b in bobSizes:
#             if not bob_dict.get(a):
#                 bob_dict[b] = 1
#         if sum_alice > sum_bob:
#             for k in sorted(bob_dict.keys()):
#                 if (sum_alice - sum_bob) % 2 == 0:
#                     if alice_dict.get(int((sum_alice - sum_bob) // 2 + k)):
#                         return [(sum_alice - sum_bob) // 2 + k, k]
#         else:
#             for k in sorted(alice_dict.keys()):
#                 if (sum_bob - sum_alice) % 2 == 0:
#                     if bob_dict.get(int((sum_bob - sum_alice) // 2 + k)):
#                         return [k, (sum_bob - sum_alice) // 2 + k]

from collections import defaultdict, deque

# class Solution:
#     def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
#         #题目实际上是最短路问题，我们可以考虑使用 BFS 来解决
#         #
#         g = [defaultdict(list), defaultdict(list)]#[从节点i出发的所有红色边, 从节点i出发的所有蓝色边]
#         for i, j in redEdges:
#             g[0][i].append(j)
#         for i, j in blueEdges:
#             g[1][i].append(j)
#         # 用来存储起点到每个节点的最短距离
#         ans = [-1] * n
#         # 存储已经搜索过的节点，以及当前边的颜色
#         vis = set()
#         # 存储当前搜索到的节点，以及当前边的颜色
#         q = deque([(0, 0), (0, 1)])
#         # 表示当前搜索的层数，即起点到当前搜索到的节点的距离
#         d = 0
#         while q:
#             for _ in range(len(q)):
#                 i, c = q.popleft()
#                 if ans[i] == -1:
#                     ans[i] = d
#                 vis.add((i, c))
#                 # c的取值只能是0或1
#                 c ^= 1
#                 for j in g[c][i]:
#                     if (j, c) not in vis:
#                         q.append((j, c))
#             d += 1
#
#         return ans


# class Solution:
#     def findSubsequences(self, nums: List[int]) -> List[List[int]]:
#         nums.append(-1000)
#         tmp_seg = []
#         i = 0
#         j = 0
#         while i < len(nums) - 1 and j < len(nums) - 2:
#             if nums[j+1] >= nums[j]:
#                 j += 1
#
#             else:
#                 tmp_seg.append(nums[i:j + 1])
#                 i = j + 1
#         tmp_seg.append(nums[i:j + 1])
#
#         res = []
#         for seg in tmp_seg:
#             for i in range(len(seg)):
#                 for j in range(i+1, len(seg)):
#                     res.append(seg[i:j+1])
#         return res

# class Solution:
#     def nextGreatestLetter(self, letters: List[str], target: str) -> str:
#         if ord(target) >= ord(letters[-1]):
#             return letters[0]
#         if ord(target) < ord(letters[0]):
#             return letters[0]
#         left = 0
#         right = len(letters) - 1
#         while left < len(letters) - 1 and right > 0 and left < right - 1:
#             if ord(letters[int((left + right)/2)]) > ord(target):
#                 right = int((left + right)/2)
#             else:
#                 left = int((left + right)/2)
#         # if left == right:
#         #     return letters[left]
#         return letters[right]


# class Solution:
#     def getFolderNames(self, names: List[str]) -> List[str]:
#         tmp = {}
#         res = []
#         for name in names:
#             if name not in tmp.keys():
#                 tmp[name] = 0
#                 res.append(name)
#             else:
#                 k = tmp[name] + 1
#                 while name + "({})".format(k) in tmp.keys():
#                     k += 1
#                 tmp[name] = k
#                 tmp[name + "({})".format(k)] = 0
#                 res.append(name + "({})".format(k))
#         # print(tmp)
#         return res
#
#
# def ceshi_corr():
#     import pandas as pd
#     import numpy as np
#     import seaborn as sns
#     from math import log
#     import matplotlib.pyplot as plt
#     y = [tmp for tmp in range(10)]
#     x1 = [log(tmp, 2) for tmp in range(1, 11)]
#     x2 = [1000* tmp**3 for tmp in range(10)]
#     x3 = [tmp for tmp in np.random.randint(0,10, 10)]
#     data = pd.DataFrame({'x1':x1, 'x2':x2, 'x3':x3, 'y':y})
#     datacorr = data.corr()
#     sns.heatmap(datacorr)
#     plt.show()
#     print(datacorr["y"])

class ToShp():
    import geopandas as gpd
    import shapefile
    from itertools import chain
    from osgeo import osr



    @staticmethod
    def to_shp(coord_flag,
               gdf_data,
               out_path='d:/polygon.shp',
               encoding='gbk',

               geometry_name='geometry'):
        '''
        功能：将geopandas导入的gdf导出为shp格式文件，
        目前支持
        polygon,MultiPolygon,BaseMultipartGeometry
        Point,MultiPoint,
        LineString,LinearRing,MultiLineString
        不支持 GeometryCollection
        主要解决geopandas导出shp文件中文乱码问题
        '''
        # 将‘geometry’列放到最后
        if coord_flag == 'geo':
            epsg_n = 4326
        else:
            epsg_n = 3857
        gdf_data = gdf_data.reindex(
            columns=(gdf_data.columns.drop(geometry_name).insert(gdf_data.shape[1], geometry_name)))
        gdf_data = gpd.GeoDataFrame(gdf_data, crs="epsg:{}".format(epsg_n), geometry=geometry_name)
        w = shapefile.Writer(out_path, encoding='gbk')
        # 列名传给shp的属性列名
        [w.field(x) for x in gdf_data.drop(columns=geometry_name).columns]
        # 循环每一行数据判断类型整理成对应的列表写入w
        for row in gdf_data.iterrows():
            geo = row[1]['geometry']
            if geo.type == 'MultiPolygon':  # 判断是否为MultiPolygon是就循环下
                exterior_z = []
                for data in geo:
                    data_a = data.exterior
                    exterior_z.append([[list(x) for x in list(data_a.coords)]])
                    for data_i in data.interiors:
                        exterior_z.append([[list(x) for x in list(data_i.coords)]])
                exterior = list(chain(*exterior_z))
                w.poly(exterior)
            elif geo.type == 'Polygon':
                def to_co(ring):
                    return [[list(x) for x in list(ring.coords)]]

                exterior = list(chain.from_iterable(map(to_co, [geo.exterior, *geo.interiors])))
                w.poly(exterior)
            elif geo.type == 'Point':
                # exterior = (geo.x, geo.y)
                w.point(geo.x, geo.y)
            elif geo.type == 'MultiPoint':
                exterior = list(map(lambda x: [x.x, x.y], geo))
                w.multipoint(exterior)
            elif geo.type == 'LineString':
                exterior = [[list(x) for x in list(geo.coords)]]
                w.line(exterior)
            elif geo.type == 'LinearRing':
                exterior = [[list(x) for x in list(geo.coords[:-1])]]
                w.line(exterior)
            elif geo.type == 'MultiLineString':
                exterior = [[list(x) for x in list(x.coords)] for x in geo]
                w.linem(exterior)
            else:
                print('错误一行')
                print(geo.type)
            w.record(*[x for x in row[1][:-1]])  # 输出每一行的内容除了geometry
        w.close()
        # 设置投影，通过.prj文件设置，需要写入一个wkt字符串
        ##gdal的GetProjection()返回的是wkt字符串，需要ImportFromWkt
        # projstr="""PROJCS["WGS_1984_UTM_zone_50N",

        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326)
        # 或 proj.ImportFromProj4(proj4str)等其他的来源
        wkt = proj.ExportToWkt()
        # 写出prj文件

        f = open(out_path.replace(".shp", ".prj"), 'w')
        if coord_flag == 'geo':
            f.write(
                'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
        else:
            f.write(
                'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
        f.close()

class Solution:
    def minNumberOfSemesters(self, n: int, relations: List[List[int]], k: int) -> int:
        rel_dic = defaultdict(list)
        indegree = [0] * n
        for [a,b] in relations:
            rel_dic[a-1].append(b-1)
            indegree[b-1] += 1
        # print(indegree)
        q = deque()
        for i, a in enumerate(indegree):

            if a == 0:
                q.append(i)
        print(q)
        res = 0
        cnt = 0
        while q :
            tmp = []
            while q and cnt < k:

                i = q.popleft()
                for j in rel_dic[i]:
                    indegree[j] -= 1
                    if indegree[j] == 0:
                        tmp.append(j)
                cnt += 1
            print(tmp)
            for j in tmp:
                q.append(j)
            res += 1
            cnt = 0


        return res

# if __name__ == '__main__':
    # ceshi_corr()

if __name__ == '__main__':
    s = Solution()
    print(s.minNumberOfSemesters(13, [[12,8],[2,4],[3,7],[6,8],[11,8],[9,4],[9,7],[12,4],[11,4],[6,4],[1,4],[10,7],[10,4],[1,7],[1,8],[2,7],[8,4],[10,8],[12,7],[5,4],[3,4],[11,7],[7,4],[13,4],[9,8],[13,8]], 9))

