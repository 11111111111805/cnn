#对数据进行最大最小规范化，是数据的范围分别落在 【0,1】和 【-1,1】区间内

class MinMaxNorm01(object):
    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        print('Min:{}, Max:{}'.format(self.min, self.max))      #用于计算数据的最大最小值，并打印

    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)        #Min-Max归一化转换公式
        return x

    def fit_transform(self, x):                                 #小调用最大最小值，然后进行MinMax归一化
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min                #将归一化数据反向转换成原始数据
        return x


class MinMaxNorm11(object):
    def __init__(self):
        pass

    def fit(self, x):
        #self.min = self.min()
        #self.max = self.max()   代码似乎有点问题，self.min/max被调用为方法而不是属性
        self.min = x.min()
        self.max = x.max()
        print('Min:{}, Max:{}'.format(self.min, self.max))

    def transform(self, x):
        x = (x - self.min) / (self.max - self.min)
        x = 2.0 * x - 1.0                                       # 数据范围【-1,1】
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = (x + 1.0) / 2.0
        x = x * (self.max - self.min) + self.min
        return x