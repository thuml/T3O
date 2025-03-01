import os

import matplotlib


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

import pandas as pd  # requires: pip install pandas
import torch
from chronos import ChronosPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = ChronosPipeline.from_pretrained(
    "./Chronos/ckpt/small",
    device_map=device,
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# forecast shape: [num_series, num_samples, prediction_length]
# print("df['#Passengers']", df["#Passengers"])
# df['#Passengers'] 0      112
# 1      118
# 2      132
# 3      129
# 4      121
#       ...
# 139    606
# 140    508
# 141    461
# 142    390
# 143    432
import numpy as np
data = torch.Tensor(df["#Passengers"][:].values).to(device)


# print("data", data)
print("data.shape", data.shape)
num_samples = 20
prediction_length = 96
forecast = pipeline.predict(
    context=data,
    prediction_length=prediction_length,
    num_samples=num_samples,
    limit_prediction_length=False,
)
print("len(df)", len(df))
print("forecast.shape", forecast.shape)
assert forecast.shape == (1, num_samples, prediction_length), f"forecast.shape={forecast.shape}"

# print(ChronosPipeline.predict.__doc__)

import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np

forecast_index = range(len(df), len(df) + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
pred = np.median(forecast[0].numpy(), axis=0)

plt.figure(figsize=(8, 4))
plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.plot(forecast_index, pred, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()
