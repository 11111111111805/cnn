import matplotlib.pyplot as plt
import pandas as pd

# 假设 data_df 是已经加载并处理好的 DataFrame
# 这里只是为了完整性，你已经有了 data_df

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))

for column in data_df.columns:
    ax.plot(data_df[column], label=column)

ax.set_title('Feature Data Visualization')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.legend()
plt.show()