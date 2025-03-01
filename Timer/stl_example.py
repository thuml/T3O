import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stl._stl import STL


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

# 创建示例时间序列数据
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
data = np.sin(2*np.pi*np.arange(365)/365) + np.random.normal(0, 0.3, 365)
ts = pd.Series(data, index=dates)

# 进行 STL 分解
# result = seasonal_decompose(ts, model='additive')  # 可以设置周期参数
result = STL(ts, period=100).fit()

# 绘制分解结果
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(ts, label='Original')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(result.trend, label='Trend')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(result.seasonal, label='Seasonal')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(result.resid, label='Residual')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
