import os

import matplotlib


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Simulated data to illustrate
np.random.seed(0)
ablation_categories = ['w/o Trimmer', 'w/o Preprocessor', 'w/o Postprocessor', 'w/o Normalizer']
num_samples = 20  # Number of samples per category

# Create repeated ablation categories
ablation = np.repeat(ablation_categories, num_samples)

# Create improvement percentages with a normal distribution for each metric
improve_percent_mean_mse = np.random.normal(loc=20, scale=5, size=len(ablation))
improve_percent_median_mse = np.random.normal(loc=25, scale=5, size=len(ablation))

# Create DataFrame
df_data_mean_by_ablation = pd.DataFrame({
    'ablation': ablation,
    'improve_percent_mean_mse': improve_percent_mean_mse,
    'improve_percent_median_mse': improve_percent_median_mse
})

# Melt the DataFrame to long-form for Seaborn
df_melted = df_data_mean_by_ablation.melt(id_vars='ablation',
                                          value_vars=['improve_percent_mean_mse', 'improve_percent_median_mse'],
                                          var_name='Metric', value_name='Improvement')

fig, ax = plt.subplots(figsize=(20, 12))

# Violin plot
sns.violinplot(ax=ax, x='ablation', y='Improvement', hue='Metric', data=df_melted, split=True, inner=None)
# Strip plot
sns.stripplot(ax=ax, x='ablation', y='Improvement', hue='Metric', data=df_melted, dodge=True, jitter=True, color='k',
              alpha=0.5)

ax.set_title('Violin and Strip Plot of Improvement by Ablation')
ax.set_xlabel('Ablation')
ax.set_ylabel('Improvement Percentage')
ax.legend(loc='upper right', title='Metric')
ax.tick_params(axis='x', rotation=75)

fig.tight_layout()
plt.show()
