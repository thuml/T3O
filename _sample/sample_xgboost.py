import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
import matplotlib


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

# 假设我们有三个不同的时序预测模型的预测结果
# model1_predictions, model2_predictions, model3_predictions 是这些模型的预测结果
# y_true 是真实的时序值

# 生成示例数据
np.random.seed(0)
n_samples = 100
model1_predictions = np.random.rand(n_samples)
model2_predictions = np.random.rand(n_samples)
model3_predictions = np.random.rand(n_samples)
y_true = np.random.rand(n_samples)

# 将每个模型的预测结果作为特征
X = np.vstack((model1_predictions, model2_predictions, model3_predictions)).T
y = y_true

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 XGBoost 进行集成
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# 进行预测
y_pred = xgb_model.predict(X_test)

# 计算 MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 进行图表展示
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True Values')
plt.plot(y_pred, label='Predictions')
plt.legend()
plt.show()
