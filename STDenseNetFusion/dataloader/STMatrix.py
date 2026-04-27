import pandas as pd
import numpy as np


class STMatrix(object):
    def __init__(self, data, timestamps, T=24, CheckComplete=True):
        super(STMatrix, self).__init__()        #调用父类构造函数
        assert len(data) == len(timestamps)     #确保传入的数据与时间戳具有相同的长度
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = timestamps         #创建Panda时间戳对象
        self.make_index()                       #根据时间戳创建索引字典，方便后续根据时间戳查找数据

    def make_index(self):
        self.get_index = dict()                 #创建空字典，存储时间戳和其对应索引的映射关系
        for i, ts in enumerate(self.pd_timestamps):     #使用 enumerate(self.pd_timestamps) 迭代时间戳列表 self.pd_timestamps，同时获取时间戳的索引 i 和时间戳 ts，返回值为 当前元素索引和当前元素值
            self.get_index[ts] = i


    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]     #通过getindex获取该时间戳在数据列表中的索引，并根据索引从数据列表data中取出对应的数据矩阵

    def check_it(self, depends):        #检查给定的时间戳依赖列表depends是否存在于数据集中
        for d in depends:               #遍历每个depends列表中的每个时间戳 d ，
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)     #import Pandas as pd
        XC = []                 #接近性数据列表
        XP = []                 #周期性数据列表
        XT = []                 #趋势性数据列表
        Y = []                  #输出数据列表
        timestamps_Y = []       #对应时间戳列表
        depends = [range(1, len_closeness+1),       #定义依赖关系列表depends，包括接近性依赖，周期性依赖，趋势性依赖，用于确定每个时间步所依赖的时间戳
                   [PeriodInterval * self.T * j for j in range(1, len_period+1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend+1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)        #根据接近性、周期性和趋势性依赖的最大长度来初始化 i，确保不超过数据集的长度
        while i < len(self.pd_timestamps):      #循环生成数据集，在 while 循环中，检查当前时间步的依赖关系是否完整，如果完整则根据依赖关系获取对应的数据，然后将数据添加到对应的数据列表中，并更新时间步和索引
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)           #将数据列表转换为数组
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y