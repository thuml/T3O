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


# 生成合成信号（示例：频率为1 Hz，相位为0.5）
t = np.linspace(0, 1, 1000)
freq = 1  # 频率
phase = 0.5  # 相位

signal = np.sin(2 * np.pi * freq * t + phase)

# 对信号进行上下翻转
flipped_signal = np.flipud(signal)

# 进行傅里叶变换
fft_signal = np.fft.fft(signal)
fft_flipped_signal = np.fft.fft(flipped_signal)

# 提取频谱幅度和相位
amp_spectrum_signal = np.abs(fft_signal)
amp_spectrum_flipped = np.abs(fft_flipped_signal)

phase_spectrum_signal = np.angle(fft_signal)
phase_spectrum_flipped = np.angle(fft_flipped_signal)

# 绘制频谱幅度和相位
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(amp_spectrum_signal)
plt.title('Amplitude Spectrum (Original)')
plt.xlabel('Frequency')

plt.subplot(2, 2, 2)
plt.plot(phase_spectrum_signal)
plt.title('Phase Spectrum (Original)')
plt.xlabel('Frequency')

plt.subplot(2, 2, 3)
plt.plot(amp_spectrum_flipped)
plt.title('Amplitude Spectrum (Flipped)')
plt.xlabel('Frequency')

plt.subplot(2, 2, 4)
plt.plot(phase_spectrum_flipped)
plt.title('Phase Spectrum (Flipped)')
plt.xlabel('Frequency')

plt.tight_layout()
plt.show()
