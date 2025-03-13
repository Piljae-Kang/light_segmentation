import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("TkAgg")

# 1) 예시 phase 신호 만들기 (길이 400)
#    선형적으로 증가하는 구간(0~100), 노이즈 구간(100~200), 다시 선형 구간(200~300), 임의 구간(300~400)...
np.random.seed(0)
phase_linear1 = np.linspace(0, 5*np.pi, 100)            # 선형 구간 A
phase_noise   = np.linspace(5*np.pi, 10*np.pi, 100) + np.random.randn(100)*2  # 노이즈 구간
phase_linear2 = np.linspace(10*np.pi, 14*np.pi, 100)    # 선형 구간 B
phase_random  = np.random.rand(100)*3 + 14*np.pi        # 임의 구간
phase = np.concatenate((phase_linear1, phase_noise, phase_linear2, phase_random))

# 2) 1차 미분(gradient) 구하기
grad = np.gradient(phase)

# 3) "선형적으로 증가"한다고 간주할 만한 slope(기울기)를 찾음
#    여기서는 단순히 전체 gradient의 "중간값" 혹은 "평균값" 등을 기준으로 삼아봄.
median_slope = np.median(grad)

# 4) 임계치(threshold)를 정해, median_slope와 얼만큼 가까우면 "선형 구간"으로 볼지 결정
threshold = 0.05  # 예시값. 실제 데이터 특성에 따라 조정해야 함

# 5) 마스크 생성: 기울기가 median_slope와 유사한(가까운) 구간은 True, 나머지는 False
mask = np.abs(grad - median_slope) < threshold

# 6) 원본 phase 신호에서, mask가 False인(즉 선형 구간이 아닌) 부분을 0으로 처리
phase_masked = np.where(mask, phase, 0)

# ------------------------------
# 시각화
# ------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(phase, label='Original Phase')
axes[0].set_ylabel('Phase')
axes[0].legend()

axes[1].plot(grad, label='Gradient')
axes[1].axhline(median_slope, color='r', linestyle='--', label='Median slope')
axes[1].set_ylabel('Gradient')
axes[1].legend()

axes[2].plot(phase_masked, label='Masked Phase', color='g')
axes[2].set_xlabel('Sample Index')
axes[2].set_ylabel('Phase (Masked)')
axes[2].legend()

plt.tight_layout()
plt.show()
