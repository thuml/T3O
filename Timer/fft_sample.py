import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

# 生成带噪声的示例时序数据
t = np.linspace(0, 5, 1000)  # 时间序列
freq = 5  # 信号的频率
signal = np.sin(2 * np.pi * freq * t)  # 信号

# 添加高斯噪声
noise = 0.5 * np.random.randn(t.size)  # 高斯噪声
noisy_signal = signal + noise  # 含噪声的信号

# 进行傅里叶变换
fft_result = np.fft.fft(noisy_signal)  # 傅里叶变换

# 获取频率轴
freqs = np.fft.fftfreq(t.size, t[1] - t[0])
print(freqs)  # 等间距的 最大100 = 1 / interval / 2 = F / 2 = Nyquist Frequency ！！！！！
print(max(freqs), len(freqs), len(noisy_signal), t.size, t[1] - t[0])

# 频谱滤波：去除高频噪声
cutoff_freq = max(abs(freqs)) / 1000  # 截止频率 cutoff_freq越小，去掉的高频越多，越平滑
fft_result[np.abs(freqs) > cutoff_freq] = 0  # 将高于截止频率的频率成分置零

# 反向傅里叶变换得到滤波后的信号
filtered_signal = np.fft.ifft(fft_result).real

# 绘制原始信号和滤波后的信号
plt.figure(figsize=(10, 6))
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.plot(t, filtered_signal, label='Filtered Signal', linewidth=2)
plt.plot(t, signal, label='Original Signal', linestyle='--', color='k', linewidth=2)
plt.title('Time Series Denoising using FFT')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
