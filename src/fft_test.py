import cv2
import numpy as np
import glob
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from sklearn.linear_model import LinearRegression

def my_hilbert(x):

    import matplotlib
    matplotlib.use("TkAgg")

    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    X = np.fft.fft(x, n=N)
    
    freq = np.fft.fftfreq(N, d=1.0)

    magnitude = np.abs(X)

    index = magnitude < np.mean(X)
    X[index] = 0.0

    plt.figure(figsize=(14, 6))
    plt.stem(freq, np.abs(X), basefmt=" ")
    plt.title('Original Signal FFT Spectrum (Positive Frequencies)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
    
    h = np.zeros(N)
    
    if N % 2 == 0:
        # 짝수 길이
        h[0] = 1          # DC 성분
        h[N//2] = 1       # Nyquist 주파수
        h[1:N//2] = 2     # 양의 주파수에 2배
    else:
        # 홀수 길이
        h[0] = 1          # DC 성분
        h[1:(N+1)//2] = 2 # 양의 주파수에 2배
    
    X_filtered = X * h
    
    x_analytic = np.fft.ifft(X_filtered, n=N)
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(x, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(x_analytic.real, label='Real Part (Original Signal)', color='blue')
    plt.plot(x_analytic.imag, label='Imaginary Part (Hilbert Transform)', color='orange')
    plt.title('Analytic Signal Components')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    amplitude_envelope = np.abs(x_analytic)
    plt.plot(amplitude_envelope, label='Amplitude Envelope', color='green')
    plt.plot(x, label='Original Signal', alpha=0.5)
    plt.title('Amplitude Envelope')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return x_analytic

def show_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # 마우스가 움직일 때
        pixel_value = img[y, x]  # (y, x) 위치의 픽셀 값
        print(f"Position: ({y}, {x}), Pixel Value: {pixel_value}")

def filtering(image, parameter=3, type=0):
    
    if type == 0:
        return median_filter(image, size=parameter)
    
    if type == 1:
        return gaussian_filter(image, sigma=parameter)
    
    if type == 2:
        return gaussian_filter1d(image, sigma=parameter, axis=1)
    
    else:
        return image

# Sigmoid 함수
def sigmoid(x, alpha=1.0):
    return 1 / (1 + np.exp(-alpha * x))


def apply_gamma(image, max_value, gamma=1.0):

    corrected = np.power(image/max_value, gamma)
    return corrected


def is_linear_phase(phase_segment, time_segment, r2_threshold=0.99):
    model = LinearRegression()
    model.fit(time_segment.reshape(-1, 1), phase_segment.reshape(-1, 1))
    r2 = model.score(time_segment.reshape(-1, 1), phase_segment.reshape(-1, 1))
    return r2 >= r2_threshold, r2


def adaptive_alpha_scaling(image):

    import matplotlib
    matplotlib.use("TkAgg")

    alpha_map = np.zeros_like(image)
    confidence_map = np.zeros_like(image)
    signal_map = np.zeros_like(image)

    for i in range(image.shape[1]):

        value = image[:, i]

        value_r = value

        analytic_signal = hilbert(value)
        analytic_signal_r = hilbert(value_r)
        analytic_signal_r = analytic_signal_r

        #analytic_signal = my_hilbert(value)
        # real_part = np.real(analytic_signal)
        # imag_part = np.imag(analytic_signal)
        amplitude_envelope = np.abs(analytic_signal)
        amplitude_envelope_r = np.abs(analytic_signal_r)

        alpha = 0.1/amplitude_envelope

        no_signal_index = amplitude_envelope < 0.01

        amplitude_envelope[no_signal_index] = 0
        #alpha[no_signal_index] = 0
        
        confidence_map[:, i] = amplitude_envelope

        masked_signal = amplitude_envelope.copy()
        masked_signal[amplitude_envelope < 0.03] = 0

        phase = np.unwrap(np.angle(analytic_signal))
        grad_phase = np.gradient(phase)

        grad_index = grad_phase < 0.13

        grad_phase[grad_index] = 0

        masked_signal[grad_index] = 10
        confidence_map[:, i] = masked_signal

        alpha[grad_index] = 5

        alpha_map[:, i] = alpha

        #################


        test_value = image[:, i]

        signal_map[:, i] = amplitude_envelope

        if i == 300:

            plt.figure(figsize=(16, 12))
            plt.subplot(2, 1, 1)
            plt.plot(np.arange(image.shape[0]), value, marker='o', linestyle='-', markersize=4)
            plt.plot(np.arange(image.shape[0]), alpha, label='Amplitude Envelope', linestyle='--')
            plt.plot(np.arange(image.shape[0]), amplitude_envelope, label='Amplitude Envelope', linestyle='--')
            plt.plot(np.arange(image.shape[0]), amplitude_envelope_r, label='Amplitude Envelope_r', linestyle='--')
            plt.plot(np.arange(image.shape[0]), masked_signal, label='masked_signal', linestyle='--')
            plt.title('Visualization of Image Column Values')
            plt.xlabel('Row Index')
            plt.ylabel('Pixel Value')

            #breakpoint()

            plt.subplot(2, 1, 2)
            #plt.plot(np.arange(image.shape[0]), alpha, label='Amplitude Envelope', linestyle='--')
            plt.plot(np.arange(image.shape[0]), phase, marker='o', linestyle='-', markersize=4)
            # plt.plot(np.arange(image.shape[0]), grad_phase, label='grad_phase', linestyle='--')
            # plt.plot(np.arange(image.shape[0]), masked_signal, label='masked_signal', linestyle='--')
            plt.title('Visualization of Image Column Phase')
            plt.xlabel('Row Index')
            plt.ylabel('Phase')

            plt.show()

            #breakpoint()
    
    
    cv2.imshow("image", image)
    cv2.imshow("confidence_map", confidence_map/np.max(confidence_map) * 5)
    cv2.imshow("alpha_map", alpha_map/np.max(alpha_map))
    cv2.imshow("signal_map", signal_map/np.max(signal_map))
    cv2.waitKey(0)

    breakpoint()

    return alpha_map, confidence_map

def fft_filtering(x, idx):
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        
        X = np.fft.fft(x, n=N)
        
        freq = np.fft.fftfreq(N, d=1.0)

        magnitude = np.abs(X)

        sorted_value = np.sort(magnitude)

        index = magnitude < np.mean(X)
        X[index] = 0.0

        new_x = np.fft.ifft(X, n=N)
        # if idx == 114:

        #     import matplotlib
        #     matplotlib.use("TkAgg")

        #     plt.figure(figsize=(14, 6))
        #     plt.stem(freq, np.abs(X), basefmt=" ")
        #     plt.title('Original Signal FFT Spectrum (Positive Frequencies)')
        #     plt.xlabel('Frequency (Hz)')
        #     plt.ylabel('Magnitude')
        #     plt.grid(True)
        #     plt.show()

        #     plt.figure(figsize=(14, 8))
    
        #     plt.subplot(3, 1, 1)
        #     plt.plot(x, label='Original Signal')
        #     plt.title('Original Signal')
        #     plt.xlabel('Sample')
        #     plt.ylabel('Amplitude')
        #     plt.legend()
        #     plt.grid(True)
            
        #     plt.subplot(3, 1, 2)
        #     plt.plot(X.real, label='Real Part (Original Signal)', color='blue')
        #     plt.title('Analytic Signal Components')
        #     plt.xlabel('Sample')
        #     plt.ylabel('Amplitude')
        #     plt.legend()
        #     plt.grid(True)
            
        #     plt.tight_layout()
        #     plt.show()
        
        return new_x

def frequency_filtering(image):

    new_image = np.zeros_like(image)

    for i in range(image.shape[1]):
        
        value = image[:, i]
        new_value = fft_filtering(value, i)
        new_image[:, i] = new_value.real

        # if i == 114:
        #     make_plot(new_image[:,i])
        #     make_plot(image[:, i])
    
    return new_image


def make_plot(value):

    import matplotlib
    matplotlib.use("TkAgg")

    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(image.shape[0]), value, marker='o', linestyle='-', markersize=4)
    plt.title('Visualization of Image Column Values')
    plt.xlabel('Row Index')
    plt.ylabel('Pixel Value')
    plt.grid(True)
    plt.show()

def is_sinusoidal(window, threshold=0.8):
    """
    주어진 윈도우가 사인파와 유사한지 평가.
    threshold: 최대 주파수 성분의 에너지 비율 기준
    """
    N = len(window)
    X = np.fft.fft(window)
    magnitude = np.abs(X)
    total_energy = np.sum(magnitude**2)
    max_energy = np.max(magnitude**2)
    
    if total_energy == 0:
        return False
    ratio = max_energy / total_energy
    return ratio >= threshold

def filter_non_sinusoidal(signal, window_size, overlap=0, threshold=0.8):
    """
    신호를 슬라이딩 윈도우로 나누어, 사인파 형태가 아닌 부분을 0으로 만듦.
    
    Parameters:
    - signal: 입력 시계열 데이터 (1D numpy array)
    - window_size: 각 윈도우의 길이
    - overlap: 윈도우 간의 중첩 길이
    - threshold: 사인파 유사도 평가 임계값
    
    Returns:
    - filtered_signal: 비사인파 구간이 0으로 처리된 신호
    """
    step = window_size - overlap
    N = len(signal)
    filtered_signal = np.copy(signal)
    
    for start in range(0, N - window_size + 1, step):
        end = start + window_size
        window = signal[start:end]
        if not is_sinusoidal(window, threshold=threshold):
            # 해당 윈도우를 0으로 설정
            filtered_signal[start:end] = 0
    
    return filtered_signal


material = "metal_shaft"
images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/1/*.png")
images_path.sort()

images = []
for image_path in images_path:
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    images.append(img)

images = np.array(images)

image = images[0] - images[2]

gaussian_image = filtering(image, parameter=1.0, type=1)
median_image = filtering(image, parameter=3, type=0)

image_repeart = np.repeat(images[2].reshape(image.shape[0], image.shape[1], 1), 3, axis=-1)

cv2.imshow("image_repeart", image_repeart)
cv2.setMouseCallback('image_repeart', show_pixel_value)
cv2.waitKey(0)

line_image = images[0].copy()
line_image[:, 300] = 1.0

make_plot(images[0][:, 300])
make_plot(images[2][:, 300])
make_plot(image[:, 300])
cv2.circle(images[2], (250,300), 2, (0, 0, 255), 1)
cv2.imshow("line_image", line_image)
cv2.imshow("images[2]", images[2])
cv2.waitKey(0)

max_image = np.maximum.reduce([images[0], images[1], images[2], images[3]])
min_image = np.minimum.reduce([images[0], images[1], images[2], images[3]])
L_d = max_image - min_image
L_g = 0.5 * min_image

original_image = np.ones_like(L_d)

image_mask = max_image < 0.05
original_image[image_mask] = 0


w_gaussian = sigmoid(20 * gaussian_image)
w_median = sigmoid(20 * median_image)
w_original = sigmoid(20 * image)


alpha_map_gaussian, confidence_map = adaptive_alpha_scaling(gaussian_image)
alpha_map_median, _ = adaptive_alpha_scaling(median_image)
alpha_map, _ = adaptive_alpha_scaling(image)

cv2.imshow("confidence_map", confidence_map/np.max(confidence_map) * 10)
cv2.waitKey(0)

adaptive_alpha_scaling_w_5_gaussian = sigmoid(5 * alpha_map_gaussian * gaussian_image)
adaptive_alpha_scaling_w_10_gaussian = sigmoid(10 * alpha_map_gaussian * gaussian_image)
adaptive_alpha_scaling_w_15_gaussian = sigmoid(15 * alpha_map_gaussian * gaussian_image)
adaptive_alpha_scaling_w_20_gaussian = sigmoid(20 * alpha_map_gaussian * gaussian_image)
adaptive_alpha_scaling_w_30_gaussian = sigmoid(30 * alpha_map_gaussian * gaussian_image)

adaptive_alpha_scaling_w_20_median = sigmoid(20 * alpha_map_median * gaussian_image)
adaptive_alpha_scaling_w_20_none = sigmoid(20 * alpha_map * gaussian_image)


# f_filtered_image = frequency_filtering(image)
# w_f_filtered_image = sigmoid(30 *f_filtered_image)

#cv2.imshow("w_f_filtered_image", w_f_filtered_image)

cv2.imshow("w_original", w_original)
cv2.imshow("w_gaussian", w_gaussian)
cv2.imshow("w_median", w_median)
cv2.imshow("adaptive_alpha_scaling_w_15", adaptive_alpha_scaling_w_15_gaussian)
cv2.imshow("adaptive_alpha_scaling_w_30", adaptive_alpha_scaling_w_30_gaussian)
cv2.setMouseCallback('w_original', show_pixel_value)
cv2.waitKey(0)

print("make plot ~")

make_plot(w_original[:, 300])
make_plot(w_gaussian[:, 300])
make_plot(w_median[:, 300])
make_plot(adaptive_alpha_scaling_w_15_gaussian[:, 300])
make_plot(adaptive_alpha_scaling_w_20_gaussian[:, 300])
make_plot(adaptive_alpha_scaling_w_20_median[:, 300])
make_plot(adaptive_alpha_scaling_w_20_none[:, 300])

fft_adaptive_alpha_scaling_w_15 = frequency_filtering(adaptive_alpha_scaling_w_15_gaussian)
# make_plot(fft_adaptive_alpha_scaling_w_15[:, 114])

cv2.imshow("fft_adaptive_alpha_scaling_w_15", fft_adaptive_alpha_scaling_w_15)
cv2.waitKey(0)


cv2.imshow("synthesized_image", L_d * adaptive_alpha_scaling_w_30_gaussian)
cv2.imshow("synthesized_image2", L_d * w_gaussian)
cv2.waitKey(0)

image = np.ones_like(L_d)

mask = L_d == 0

image[mask] = 0

cv2.imshow("synthesized_image", original_image * adaptive_alpha_scaling_w_30_gaussian)
cv2.imshow("synthesized_image2", original_image * w_gaussian)
cv2.waitKey(0)

make_plot((max_image * adaptive_alpha_scaling_w_30_gaussian)[:, 330])
make_plot((max_image * w_gaussian)[:, 330])

alpha_map_gaussian_filtering = filtering(alpha_map_gaussian, 1.0, type=1)

adaptive_alpha_scaling_w_30_gaussian = sigmoid(50 * alpha_map_gaussian_filtering * gaussian_image)

adaptive_alpha_scaling_w_30_gaussian2 = sigmoid(50 * filtering(alpha_map_gaussian * gaussian_image, 1.0, type=1))

cv2.imshow("adaptive_alpha_scaling_w_30_gaussian", adaptive_alpha_scaling_w_30_gaussian)
cv2.imshow("adaptive_alpha_scaling_w_30_gaussian2", adaptive_alpha_scaling_w_30_gaussian)
cv2.imshow("adaptive_alpha_scaling_w_30", adaptive_alpha_scaling_w_30_gaussian)
cv2.waitKey(0)


c_image = cv2.imread("/media/piljae/X31/light_segmentation/src/adaptive_alpha_direct_16_pattern/gaussian_filtering/alpha_29_beta_1/0.png")
c_image = c_image/255.0

make_plot(adaptive_alpha_scaling_w_30_gaussian[:, 114])
make_plot(c_image[:, 114])