import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

# 准备时间序列数据
data = {
    'time': [1, 2, 3, 4, 5],
    'value': [10, 11, 12, 13, 14],
    'id': ['A', 'A', 'A', 'A', 'A']
}
timeseries_df = pd.DataFrame(data)

# 提取特征
params = ComprehensiveFCParameters()
features = extract_features(timeseries_df, column_id='id', column_sort='time', default_fc_parameters=params)

# 打印提取的特征
print(features)
