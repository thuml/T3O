import os

import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

import matplotlib


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

# SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
# PDT = 20  # prediction length: any positive integer
# CTX = 200  # context length: any positive integer
# PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
# BSZ = 32  # batch size: any positive integer
# TEST = 100  # test set length: any positive integer


SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 96  # prediction length: any positive integer
CTX = 96  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 1  # batch size: any positive integer
TEST = 96  # test set length: any positive integer

# Read data into pandas DataFrame
url = (
    "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
)
df = pd.read_csv('./AnyTransform/ts_wide.csv', index_col=0, parse_dates=True)
# 只保留第一列和A列
df = df[['A']]
print(df)

# Convert into GluonTS dataset
ds = PandasDataset(dict(df))
print("len(ds)", len(ds))

# Split into train/test set
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set

# Construct rolling window evaluation

test_data = test_template.generate_instances(
    # prediction_length=PDT,  # number of time steps for each prediction
    # windows=TEST // PDT,  # number of windows in rolling window evaluation
    # distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows

    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)

# Prepare pre-trained model by downloading model weights from huggingface hub
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it)

print('forecast', forecast)
print('forecast.samples', forecast.samples)
print('forecast.samples.shape', forecast.samples.shape)
print('forecast.quantile(0.5)', forecast.quantile(0.5))
# exit()

plot_single(
    inp,
    label,
    forecast,
    context_length=CTX,
    name="pred",
    show_label=True,
)
plt.show()
