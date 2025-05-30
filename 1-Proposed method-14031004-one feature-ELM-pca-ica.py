"""
Windowing method in eeg data how can implement? Write code in tf and keras for eeg data: x(168,118,400) ,y(168) .Apply windowing method, extract feature after band pass filter applyed on mu and beta rythms then use the FBCSP feature extraction method . Use a feature selection method to select best informative features. After feature extraction and selection , use a classification deep learning model such as CNN for classification. 
Steps Overview:
Windowing: Segment the EEG signals into fixed-sized overlapping windows.
Band-pass Filtering: Apply band-pass filters for mu (8-13 Hz) and beta (13-30 Hz) rhythms.
Feature Extraction using FBCSP: Extract spatial features using Filter Bank Common Spatial Patterns (FBCSP).
Feature Selection: Select the most informative features.
Classification: Train a CNN on the selected features for classification.
Python Code Implementation"""
#%%
# Step 1: Setup and Imports
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import mne

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid band: low={lowcut}Hz, high={highcut}Hz exceeds filter limits for fs={fs}")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def build_simple_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(16, (1, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((1, 2)),
        layers.Conv2D(32, (1, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def select_best_rhythm(eeg_data, labels, fs=100):
    """
    eeg_data: shape (samples, channels, time_points)
    labels: binary labels, shape (samples,)
    fs: sampling rate (Hz)
    """
    rhythms = {
        'mu': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    best_acc = 0
    best_band = None

    for name, (low, high) in rhythms.items():
        if high >= fs / 2:
            print(f"Skipping {name} band: highcut {high} >= Nyquist ({fs/2})")
            continue

        try:
            # Bandpass filter each epoch and channel
            filtered = np.array([
                [bandpass_filter(channel, low, high, fs) for channel in epoch]
                for epoch in eeg_data
            ])

            # Prepare input for CNN
            X = filtered[..., np.newaxis]  # (samples, channels, time, 1)
            y = np.array(labels)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

            # Build and train CNN
            model = build_simple_cnn(X_train.shape[1:])
            model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

            # Evaluate
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"{name} band accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_band = name

        except Exception as e:
            print(f"Error processing {name} band: {e}")
            continue

    return best_band



import numpy as np
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import mne
from scipy import signal
from scipy import stats

def apply_pca_denoise(X, n_components=None, plot_results=True):
    """
    Apply PCA for dimensionality reduction and noise removal on EEG data
    
    Parameters:
    X : 3D array with shape (samples, channels, time points)
    n_components : Number of principal components to keep (if None, auto-selected)
    plot_results : Whether to plot the results
    
    Returns:
    X_reconstructed : Reconstructed data with shape (samples, channels, time points)
    """
    # First reshape the data
    samples, channels, time_points = X.shape
    X_reshaped = X.reshape(samples, channels * time_points)
    
    # Determine number of components if not specified
    if n_components is None:
        # Calculate cumulative variance for automatic component selection
        temp_pca = PCA()
        temp_pca.fit(X_reshaped)
        explained_variance_ratio = temp_pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Select number of components to cover 50% variance
        n_components = np.argmax(cumulative_variance >= 0.50) + 1
        print(f"Selected number of PCA components: {n_components} (50% variance)")

    # Adjust n_components if it exceeds allowed limits
    n_components = min(n_components, samples, channels * time_points)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_reshaped)
    
    # Reconstruct data
    X_reconstructed_flat = pca.inverse_transform(X_pca)
    X_reconstructed = X_reconstructed_flat.reshape(samples, channels, time_points)
    
    # Plot results
    if plot_results:
        plt.figure(figsize=(15, 10))
        
        # Explained variance plot
        plt.subplot(2, 2, 1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Variance vs. Number of Components')
        plt.grid(True)
        
        # Compare original vs reconstructed signal for one sample
        sample_idx = 0  # First sample
        channel_idx = 0  # First channel
        
        plt.subplot(2, 2, 2)
        plt.plot(X[sample_idx, channel_idx], label='Original Signal')
        plt.plot(X_reconstructed[sample_idx, channel_idx], label='Reconstructed Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Original vs Reconstructed Signal (Sample {sample_idx}, Channel {channel_idx})')
        plt.legend()
        plt.grid(True)
        
        # Plot first few principal components
        plt.subplot(2, 2, 3)
        n_components_to_plot = min(5, n_components)
        for i in range(n_components_to_plot):
            component = pca.components_[i].reshape(channels, time_points)
            plt.plot(component[0], label=f'Component {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('First Few Principal Components')
        plt.legend()
        plt.grid(True)
        
        # Plot removed noise (difference between original and reconstructed)
        plt.subplot(2, 2, 4)
        noise = X[sample_idx, channel_idx] - X_reconstructed[sample_idx, channel_idx]
        plt.plot(noise)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Removed Noise')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    return X_reconstructed, pca



def apply_ica_denoise(X, n_components=None, plot_results=True):
    """
    اعمال ICA برای کاهش بعد و حذف نویز روی داده‌های EEG
    
    پارامترها:
    X : آرایه سه بعدی با شکل (samples, channels, time points)
    n_components : تعداد مؤلفه‌های مستقل (اگر None باشد، به صورت خودکار انتخاب می‌شود)
    plot_results : آیا نتایج به صورت نمودار نمایش داده شود؟
    
    خروجی:
    X_reconstructed : داده‌های بازسازی شده با حذف مؤلفه‌های نویزی
    """
    samples, channels, time_points = X.shape
    
    # اگر تعداد مؤلفه‌ها مشخص نشده، از تعداد کانال‌ها استفاده می‌کنیم
    if n_components is None:
        n_components = channels
    
    # برای هر نمونه ICA را جداگانه اعمال می‌کنیم
    X_reconstructed = np.zeros_like(X)
    
    for s in range(samples):
        # تبدیل داده به فرمت MNE - باید به شکل (channels, time_points) باشد
        data = X[s]  # این داده از قبل به شکل (channels, time_points) است
        
        # ساخت یک شیء Info ساده برای MNE
        sfreq = 100  # فرکانس نمونه‌برداری فرضی
        ch_names = [f'EEG{i+1}' for i in range(channels)]
        ch_types = ['eeg'] * channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # ساخت شیء Raw
        raw = mne.io.RawArray(data, info)
        
        # اعمال ICA
        ica = ICA(n_components=n_components, random_state=42)
        ica.fit(raw)
        
        # شناسایی خودکار مؤلفه‌های نویزی
        ica_sources = ica.get_sources(raw).get_data()
        kurtosis = np.abs(stats.kurtosis(ica_sources, axis=1))
        skewness = np.abs(stats.skew(ica_sources, axis=1))
        
        # مؤلفه‌هایی که کورتوسیس یا چولگی آنها بیشتر از آستانه است به عنوان نویز در نظر گرفته می‌شوند
        kurt_threshold = np.percentile(kurtosis, 75)  # مقدار آستانه برای کورتوسیس
        skew_threshold = np.percentile(skewness, 75)  # مقدار آستانه برای چولگی
        
        noise_idx = np.where((kurtosis > kurt_threshold) | (skewness > skew_threshold))[0]
        
        # حذف مؤلفه‌های نویزی و بازسازی سیگنال
        ica.exclude = noise_idx
        clean_raw = raw.copy()
        ica.apply(clean_raw)
        
        # ذخیره سیگنال تمیز شده (بدون عوض کردن ابعاد)
        X_reconstructed[s] = clean_raw.get_data()
        
        # نمایش نتایج برای اولین نمونه
        if plot_results and s == 0:
            plt.figure(figsize=(15, 10))
            
            # نمایش سیگنال اصلی
            plt.subplot(3, 1, 1)
            plt.plot(data[:, 0])  # نمایش اولین کانال
            plt.title('سیگنال اصلی (کانال اول)')
            plt.grid(True)
            
            # نمایش مؤلفه‌های ICA
            plt.subplot(3, 1, 2)
            plt.plot(ica_sources.T)
            plt.title('مؤلفه‌های مستقل (ICA)')
            plt.grid(True)
            
            # نمایش سیگنال تمیز شده
            plt.subplot(3, 1, 3)
            plt.plot(clean_raw.get_data()[0])  # نمایش اولین کانال
            plt.title('سیگنال تمیز شده (کانال اول)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # نمایش توپوگرافی مؤلفه‌های ICA (در صورت امکان)
            # ica.plot_components()
            
            # نمایش مؤلفه‌های حذف شده
            # print(f"مؤلفه‌های شناسایی شده به عنوان نویز: {noise_idx}")
    
    return X_reconstructed

def apply_combined_pca_ica(X, pca_components=None, ica_components=None, plot_results=True):
    """
    اعمال ترکیبی PCA و سپس ICA برای کاهش بعد و حذف نویز
    
    پارامترها:
    X : آرایه سه بعدی با شکل (samples, channels, time points)
    pca_components : تعداد مؤلفه‌های PCA
    ica_components : تعداد مؤلفه‌های ICA
    plot_results : آیا نتایج به صورت نمودار نمایش داده شود؟
    
    خروجی:
    X_reconstructed : داده‌های بازسازی شده پس از اعمال PCA و ICA
    """
    # ابتدا PCA اعمال می‌شود
    X_pca, pca_model = apply_pca_denoise(X, n_components=pca_components, plot_results=plot_results)
    
    # سپس ICA روی داده‌های PCA اعمال می‌شود
    X_ica = apply_ica_denoise(X_pca, n_components=ica_components, plot_results=plot_results)
    
    return X_ica


def apply_car(data):
    car_filtered = data - np.mean(data, axis=1, keepdims=True)
    return car_filtered

# Step 2: Band-pass Filtering
def butter_bandpass(lowcut, highcut, fs, order=4):# 4
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data, axis=-1)

# Step 3: Windowing
def apply_windowing(data, window_size, step_size):
    num_samples, num_channels, num_timepoints = data.shape
    windows = []
    for i in range(0, num_timepoints - window_size + 1, step_size):
        window = data[:, :, i:i+window_size]
        windows.append(window)
    return np.array(windows)  # Shape: (num_windows, num_samples, num_channels, window_size)

# Step 4: FBCSP Feature Extraction
def fbcsp_features(data, labels, num_components=1):
    from mne.decoding import CSP

    num_samples, num_channels, num_timepoints = data.shape
    data_flattened = data.reshape(num_samples, num_channels, -1)

    csp = CSP(n_components=num_components)
    transformed_data = csp.fit_transform(data_flattened, labels)
    return transformed_data  # Shape: (num_samples, num_components)

# Step 5: Feature Selection
def select_features(features, labels, num_features):
    selector = SelectKBest(score_func=mutual_info_classif, k=num_features)
    selected_features = selector.fit_transform(features, labels)
    return selected_features

# Step 6: EEGNet Model for Classification

def build_eegnet(input_shape, num_classes):
    model = models.Sequential()

    # First Conv Layer (1D Conv)
    model.add(layers.Conv2D(8, (1, 4), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Dropout(0.25))

    # Second Conv Layer (1D Conv)
    model.add(layers.Conv2D(16, (1, 4), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Dropout(0.25))

    # Depthwise Separable Conv Layer (for EEG data)
    model.add(layers.DepthwiseConv2D((1, 4), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Dropout(0.25))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Fully Connected Layer (Dense)
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Dropout(0.5))

    # Output Layer (Softmax activation for classification)
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def extreme_learning_machine(X_train, y_train, X_test, num_hidden_neurons):
    """
    Extreme Learning Machine (ELM) Classification
    """
    # Random initialization of input weights and bias
    input_weights = np.random.randn(X_train.shape[1], num_hidden_neurons)
    bias = np.random.randn(num_hidden_neurons)
    
    # Hidden layer output
    H = np.maximum(0, np.dot(X_train, input_weights) + bias)  # ReLU activation
    
    # Output weights using Moore-Penrose pseudoinverse
    output_weights = np.linalg.pinv(H).dot(y_train)
    
    # Test phase
    H_test = np.maximum(0, np.dot(X_test, input_weights) + bias)
    y_pred = H_test.dot(output_weights)
    
    return y_pred

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics
    """
    y_pred_binary = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    error = 1 - acc

    return {
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'MCC': mcc,
        'F1-Score': f1,
        'Error': error,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mne

def plot_eeg_sample(eeg_data, sample_idx, channel_idx, text):
    """
    Plots a specific sample and channel from EEG data.

    Parameters:
        eeg_data (numpy.ndarray): EEG data in shape (samples, channels, time points).
        sample_idx (int): Index of the sample to plot.
        channel_idx (int): Index of the channel to plot.

    Returns:
        None: Displays the plot.
    """
    # Check if the indices are valid
    if sample_idx >= eeg_data.shape[0]:
        raise ValueError(f"Sample index {sample_idx} is out of bounds for EEG data with {eeg_data.shape[0]} samples.")
    if channel_idx >= eeg_data.shape[1]:
        raise ValueError(f"Channel index {channel_idx} is out of bounds for EEG data with {eeg_data.shape[1]} channels.")
    
    # Extract the data for the specified sample and channel
    data_to_plot = eeg_data[sample_idx, channel_idx, :]
    
    # Plot the data
    plt.figure(figsize=(10, 4))
    plt.plot(data_to_plot, label=f"Sample {sample_idx}, Channel {channel_idx}")
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.title(text)
    plt.legend()
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_heatmap(f, title="Feature Heatmap"):
    """
    Plots a heatmap for extracted features.

    Parameters:
        f (numpy.ndarray): Feature data of shape (samples, features).
        title (str): Title of the heatmap (default: "Feature Heatmap").
    """
    # Validate input
    if len(f.shape) != 2:
        raise ValueError("Input feature data must be a 2D array with shape (samples, features).")
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(f, cmap="viridis", cbar=True, annot=False, xticklabels=False, yticklabels=False)
    
    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Samples", fontsize=12)
    plt.tight_layout()
    plt.show()



def plot_results(history, best_fold):
    """
    Plot training/validation accuracy and loss for all folds
    """
    folds = [h['fold'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    
    plt.figure(figsize=(12, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(folds, train_acc, label='Train Accuracy', marker='o')
    plt.plot(folds, val_acc, label='Validation Accuracy', marker='o')
    plt.axvline(x=best_fold['fold'], color='r', linestyle='--', label='Best Fold')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(folds, train_loss, label='Train Loss', marker='o')
    plt.plot(folds, val_loss, label='Validation Loss', marker='o')
    plt.axvline(x=best_fold['fold'], color='r', linestyle='--', label='Best Fold')
    plt.title('Training and Validation Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
        
# Step 7: Data Pipeline and Training

# Example parameters
fs = 100  # Sampling frequency
window_size = 200  # window_size = 200(800 ms window)
step_size =   50  # step_size = 50(200 ms step)
low_mu, high_mu = 8, 13
low_beta, high_beta = 13, 30
num_features = 10
num_classes=2
num_hidden_neurons=15

# EEG data (x, y)
from PLIB02 import get_dataset
# Create a list of dataset files
dataset_files=['data_aa', # 0
               'data_al', # 1
               'data_av', # 2
               'data_aw', # 3
               'data_ay'] # 4

no=0 # number of database file
dataset_name=dataset_files[no]
x,y,z=get_dataset(dataset_name+".mat")

classes={ 'data_aa':(80,88  ),
          'data_al':(112,112),
          'data_av':(42,42  ),
          'data_aw':(30,26  ),
          'data_ay':(18,10  )
          
                       }
c0 = min(classes[dataset_name]) 
c1 = max(classes[dataset_name])
    
c0,c1=classes[dataset_name]
x1=x[0:c0,] # class 0 data
x2=x[c0:c0+c1,] # class 1 data

y1=y[0:c0]   # class 0 label
y2=y[c0:c0+c1] # class 0 label

print(F"Test for Dataset ({dataset_name})")


# best_rhythm_x1 = select_best_rhythm(x1, y1, fs=100)
# print("\n Most informative rhythm x1:", best_rhythm_x1)

# best_rhythm_x2 = select_best_rhythm(x2, y2, fs=100)
# print("\n Most informative rhythm x2:", best_rhythm_x2)
# process x1 data

# Apply CAR filter
car_filtered = apply_car(x1)
x1=car_filtered
x1,x11=apply_pca_denoise(x1, n_components=None, plot_results=True)
# x1=apply_ica_denoise(x1, n_components=None, plot_results=True)
# x1=apply_combined_pca_ica(x1, pca_components=None, ica_components=None, plot_results=True)
# Standardize each channel across the samples and time points (x)
mean = np.mean(x1, axis=0, keepdims=True)
std = np.std(x1, axis=0, keepdims=True)
x1 = (x1 - mean) / std
# Normalize each channel across the samples and time points to range [0, 1] (x)
x_min = np.min(x1, axis=0, keepdims=True)
x_max = np.max(x1, axis=0, keepdims=True)
x1 = (x1 - x_min) / (x_max - x_min)

# Band-pass filtering
x1_mu = bandpass_filter(x1, low_mu, high_mu, fs)
x1_beta = bandpass_filter(x1, low_beta, high_beta, fs)

# Windowing
x1_windows_mu = apply_windowing(x1_mu, window_size, step_size)
x1_windows_beta = apply_windowing(x1_beta, window_size, step_size)

# Combine filtered data
x1_combined = np.concatenate([x1_windows_mu, x1_windows_beta], axis=-1)  # Shape: (num_windows, num_samples, num_channels, 2*window_size)

# Flatten windows and extract features using FBCSP
num_windows, num_samples, num_channels, num_timepoints = x1_combined.shape
x1_combined_flat = x1_combined.reshape(-1, num_channels, num_timepoints)
y1_repeated = np.repeat(y1, num_windows)  # Labels for all windows

# process x2 data

# Apply CAR filter
x2 = apply_car(x2)
x2,x22=apply_pca_denoise(x2, n_components=None, plot_results=True)
# x2=apply_ica_denoise(x2, n_components=None, plot_results=True)
# x2=apply_combined_pca_ica(x2, pca_components=None, ica_components=None, plot_results=True)
# Standardize each channel across the samples and time points (x)
mean = np.mean(x2, axis=0, keepdims=True)
std = np.std(x2, axis=0, keepdims=True)
x2 = (x2 - mean) / std
# Normalize each channel across the samples and time points to range [0, 1] (x)
x_min = np.min(x2, axis=0, keepdims=True)
x_max = np.max(x2, axis=0, keepdims=True)
x2 = (x2 - x_min) / (x_max - x_min)

# Band-pass filtering
x2_mu   = bandpass_filter(x2, low_mu, high_mu, fs)
x2_beta = bandpass_filter(x2, low_beta, high_beta, fs)

# Windowing
x2_windows_mu = apply_windowing(x2_mu, window_size, step_size)
x2_windows_beta = apply_windowing(x2_beta, window_size, step_size)

# Combine filtered data
x2_combined = np.concatenate([x2_windows_mu, x2_windows_beta], axis=-1)  # Shape: (num_windows, num_samples, num_channels, 2*window_size)

# Flatten windows and extract features using FBCSP
num_windows, num_samples, num_channels, num_timepoints = x2_combined.shape
x2_combined_flat = x2_combined.reshape(-1, num_channels, num_timepoints)
y2_repeated = np.repeat(y2, num_windows)  # Labels for all windows


x_combined_flat=np.concatenate((x1_combined_flat,x2_combined_flat), axis=0)
y_repeated=np.concatenate((y1_repeated,y2_repeated), axis=0)

fbcsp_features_data = fbcsp_features(x_combined_flat, y_repeated)

# Feature selection
# selected_features = select_features(fbcsp_features_data, y_repeated, num_features=num_features)
fbcsp_features_data1=fbcsp_features_data[0:x1_combined_flat.shape[0]]
fbcsp_features_data2=fbcsp_features_data[x1_combined_flat.shape[0]:]

selected_features=fbcsp_features_data
selected_features1=fbcsp_features_data1
selected_features2=fbcsp_features_data2


# Standardize each channel across the samples and time points (x)
mean = np.mean(selected_features, axis=0, keepdims=True)
std = np.std(selected_features, axis=0, keepdims=True)
selected_features = (selected_features - mean) / std

# Normalize each channel across the samples and time points to range [0, 1] (x)
x_min = np.min(selected_features, axis=0, keepdims=True)
x_max = np.max(selected_features, axis=0, keepdims=True)
selected_features = (selected_features - x_min) / (x_max - x_min)


# Perform 10-fold cross-validation with ELM
k_folds=10
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []
best_accuracy = 0
best_fold_data = None
history = []  # Store training/validation metrics for plotting
# Initialize confusion matrices for each class
conf_matrix_class_0 = np.zeros((2, 2), dtype=int)
conf_matrix_class_1 = np.zeros((2, 2), dtype=int)

for fold, (train_idx, test_idx) in enumerate(kfold.split(selected_features, y_repeated), 1):
    # Split into training and testing
    X_train, X_test = selected_features[train_idx], selected_features[test_idx]
    y_train, y_test = y_repeated[train_idx], y_repeated[test_idx]
    
    # Further split training data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    # Train ELM model
    
    y_pred = extreme_learning_machine(X_train, y_train, X_test, num_hidden_neurons)
    y_val_pred = extreme_learning_machine(X_train, y_train, X_val, num_hidden_neurons)
    
    # Calculate metrics   
    test_metrics = calculate_metrics(y_test, y_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    fold_results.append(test_metrics)
    
    # print fold metrics
    print("*******************")
    print(f"\nFold {fold}")
    print("Accuracy:",test_metrics['Accuracy'])
    print("Sensitivity:",test_metrics['Sensitivity'])
    print("Specificity:",test_metrics['Specificity'])
    print("MCC:",test_metrics['MCC'])
    print("F1-Score:",test_metrics['F1-Score'])
    print("Error:",test_metrics['Error'])
    print("TP:",test_metrics['TP'])
    print("FP:",test_metrics['FP'])
    print("TN:",test_metrics['TN'])
    print("FN:",test_metrics['FN'])
    
    # Overall confusion matrix
    # [[TP, FP],
    # [FN, TN]]
    y_pred = (y_pred >= 0.5).astype(int)
    
    # Confusion matrix for class 0
    class_0_true = (y_test == 0).astype(int)
    class_0_pred = (y_pred == 0).astype(int)
    conf_matrix_class_0 += confusion_matrix(class_0_true, class_0_pred)

    # Confusion matrix for class 1
    class_1_true = (y_test == 1).astype(int)
    class_1_pred = (y_pred == 1).astype(int)
    conf_matrix_class_1 += confusion_matrix(class_1_true, class_1_pred)
    
    print("*******************")
    
    # Store metrics for plotting
    history.append({
        'fold': fold,
        'train_acc': test_metrics['Accuracy'],
        'val_acc': val_metrics['Accuracy'],
        'train_loss': test_metrics['Error'],
        'val_loss': val_metrics['Error']
    })
    
    # Save best fold
    if test_metrics['Accuracy'] > best_accuracy:
        best_accuracy = test_metrics['Accuracy']
        best_fold_data = {
            'X_test': X_test,
            'y_test': y_test,
            'metrics': test_metrics,
            'model_weights': (X_train.shape[1], num_hidden_neurons)  # Store model params
        }
        

# Print and save best fold details
print("\nBest Fold:")
for metric, value in best_fold_data['metrics'].items():
    print(f"{metric}: {value}")

acc_sum=0
for i in range(len(fold_results)):
    acc_sum=acc_sum+fold_results[i].get("Accuracy")
acc_avrege=acc_sum/len(fold_results) 
print("\nAverage Accuracy:",acc_avrege)
       
sen_sum=0
for i in range(len(fold_results)):
    sen_sum=sen_sum+fold_results[i].get("Sensitivity")
sen_avrege=sen_sum/len(fold_results)
print("\nAverage Sensitivity:",sen_avrege)

spe_sum=0
for i in range(len(fold_results)):
    spe_sum=spe_sum+fold_results[i].get("Specificity")
spe_avrege=spe_sum/len(fold_results)
print("\nAverage Specificity:",spe_avrege)

mcc_sum=0
for i in range(len(fold_results)):
    mcc_sum=mcc_sum+fold_results[i].get("MCC")
mcc_avrege=mcc_sum/len(fold_results)
print("\nAverage MCC:",mcc_avrege)

f1_sum=0
for i in range(len(fold_results)):
    f1_sum=f1_sum+fold_results[i].get("F1-Score")
f1_avrege=f1_sum/len(fold_results)
print("\nAverage F1-Score:",f1_avrege)

er_sum=0
for i in range(len(fold_results)):
    er_sum=er_sum+fold_results[i].get("Error")
er_avrege=er_sum/len(fold_results)
# print("\nAverage Error:",er_avrege)
print(f"\nAverage Error:{er_avrege:.4e}")

tp_sum=0
for i in range(len(fold_results)):
    tp_sum=tp_sum+fold_results[i].get("TP")
tp_avrege=tp_sum/len(fold_results)
print("\nAverage TP:",tp_avrege)

fp_sum=0
for i in range(len(fold_results)):
    fp_sum=fp_sum+fold_results[i].get("FP")
fp_avrege=fp_sum/len(fold_results)
print("\nAverage FP:",fp_avrege)

tn_sum=0
for i in range(len(fold_results)):
    tn_sum=tn_sum+fold_results[i].get("TN")
tn_avrege=tn_sum/len(fold_results)
print("\nAverage TN:",tn_avrege)

fn_sum=0
for i in range(len(fold_results)):
    fn_sum=fn_sum+fold_results[i].get("FN")
fn_avrege=fn_sum/len(fold_results)
print("\nAverage FN:",fn_avrege)

# Average the confusion matrices across folds
conf_matrix_class_0_avg = conf_matrix_class_0 / kfold.get_n_splits()
conf_matrix_class_1_avg = conf_matrix_class_1 / kfold.get_n_splits()

# Report the averaged confusion matrices
print("\nAverage Confusion Matrix for Class 0 (as positive class):")
print(conf_matrix_class_0_avg)
print("\nAverage Confusion Matrix for Class 1 (as positive class):")
print(conf_matrix_class_1_avg)
print()        
        

# Simulated EEG data
np.random.seed(42)

# Plot a specific sample and channel
plot_eeg_sample(x1, sample_idx=0, channel_idx=5, text='Raw EEG signal-Right Hand')
plot_eeg_sample(x2, sample_idx=0, channel_idx=5, text='Raw EEG signal-Right Foot')

plot_eeg_sample(x1_combined_flat, sample_idx=0, channel_idx=5, text='CAR Filtered and mu and beta band passed EEG signal of Right Hand')
plot_eeg_sample(x2_combined_flat, sample_idx=0, channel_idx=5, text='CAR Filtered and mu and beta band passed EEG signal of Right Foot')

# Plot the heatmap
plot_feature_heatmap(fbcsp_features_data, title="Heatmap of Extracted Features")