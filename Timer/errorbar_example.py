import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

# 创建示例数据
x = np.linspace(0, 10, 20)
y = np.sin(x)
# 创建示例误差，这里使用随机生成的误差值
y_err = 0.2 * np.random.rand(20)

# 绘制带有误差线的折线图
x_names = [str(i) for i in range(20)]
plt.errorbar(x_names, y, yerr=y_err, fmt='-o', label='data with error', elinewidth=2)

# 添加图形元素和标签
plt.title('Errorbar Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# 显示图形
plt.show()
