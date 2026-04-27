import matplotlib.pyplot as plt
plt.figure()  # 在matplotlib创建一个新的图形对象
# 绘制预测值和真实值的曲线图，并乘以缩放因子还原为初始值
x = range(10)
y= range(10)
plt.plot(x,y, 'r-', label='Predicted')

plt.legend(loc='upper right')
plt.show()