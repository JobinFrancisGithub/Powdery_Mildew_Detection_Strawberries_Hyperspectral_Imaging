import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.signal import savgol_filter
# ===========================
# Load your preprocessed data
def apply_snv(X):
    """Standard Normal Variate"""
    return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

def apply_msc(X):
    """Multiplicative Scatter Correction"""
    ref = np.mean(X, axis=0)
    X_msc = np.empty_like(X)
    for i in range(X.shape[0]):
        fit = np.polyfit(ref, X[i], 1, full=False)
        X_msc[i] = (X[i] - fit[1]) / fit[0]
    return X_msc

def apply_sgs(X, window_length=11, polyorder=2):
    """Savitzky-Golay Smoothing"""
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, axis=1)


def plot_pcs(X, y, title="PCA Score Plot"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    for label, marker, color in zip([0, 1], ['o', '+'], ['green', 'brown']):
        plt.scatter(
            X_pca[y == label, 0],
            X_pca[y == label, 1],
            label="Healthy" if label == 0 else "Mildew",
            alpha=0.7,
            marker=marker,
            color=color
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ========================================
# 2. Load Spectral Data
# ========================================
healthy1 = np.load(r'D:\Python_Codes_N_Results_Folder\Results\Spectrum_Strawberry_Folder\2D_array_spectra\2D_array_healthyleaves_spectra\2D_array_leavesspectra_h1.npy')  # Shape: (N1, 164)
healthy2 = np.load(r'D:\Python_Codes_N_Results_Folder\Results\Spectrum_Strawberry_Folder\2D_array_spectra\2D_array_healthyleaves_spectra\2D_array_leavesspectra_h2.npy')
healthy3 = np.load(r'D:\Python_Codes_N_Results_Folder\Results\Spectrum_Strawberry_Folder\2D_array_spectra\2D_array_healthyleaves_spectra\2D_array_leavesspectra_h3.npy')

mildew1 = np.load(r'D:\Python_Codes_N_Results_Folder\Results\Spectrum_Strawberry_Folder\2D_array_spectra\2D_array_mw_affected_leaves_spectra\2D_array_leavesspectra_mw1.npy')  # Shape: (N2, 164)
mildew2 = np.load(r'D:\Python_Codes_N_Results_Folder\Results\Spectrum_Strawberry_Folder\2D_array_spectra\2D_array_mw_affected_leaves_spectra\2D_array_leavesspectra_mw2.npy')  # Shape: (N2, 164)
mildew3 = np.load(r'D:\Python_Codes_N_Results_Folder\Results\Spectrum_Strawberry_Folder\2D_array_spectra\2D_array_mw_affected_leaves_spectra\2D_array_leavesspectra_mw3.npy')  # Shape: (N2, 164)
mildew4 = np.load(r'D:\Python_Codes_N_Results_Folder\Results\Spectrum_Strawberry_Folder\2D_array_spectra\2D_array_mw_affected_leaves_spectra\2D_array_leavesspectra_mw4.npy')  # Shape: (N2, 164)


healthy = np.vstack((healthy1, healthy2, healthy3))[:, :145]
mildew = np.vstack((mildew1, mildew2, mildew3, mildew4))[:, :145]

print(f"Healthy shape: {healthy.shape}, Mildew shape: {mildew.shape}")
#exit()

X_raw = np.vstack((healthy, mildew))
y = np.concatenate((np.zeros(healthy.shape[0]), np.ones(mildew.shape[0])))
print(X_raw.shape)

#========================================
# Split data into classes
X_healthy = X_raw[y == 0]
print(X_healthy.shape)
X_mildew = X_raw[y == 1]
print(X_mildew.shape)

# Fit IsolationForest separately
iso_healthy = IsolationForest(contamination=0.05, random_state=42)
mask_healthy = iso_healthy.fit_predict(X_healthy) == 1

iso_mildew = IsolationForest(contamination=0.05, random_state=42)
mask_mildew = iso_mildew.fit_predict(X_mildew) == 1

# Filter each class
X_healthy_filtered = X_healthy[mask_healthy]
print(X_healthy_filtered.shape)
X_mildew_filtered = X_mildew[mask_mildew]
print(X_mildew_filtered.shape)


# Recombine
X_filtered = np.vstack([X_healthy_filtered, X_mildew_filtered])
y = np.concatenate([np.zeros(len(X_healthy_filtered), dtype=int),
                    np.ones(len(X_mildew_filtered), dtype=int)])

print("New shapes:")
print("X_filtered:", X_filtered.shape)
print("y:", y.shape)

#exit()
# ========================================
# 3. Visualization of Preprocessing
# ========================================

def plot_preprocessing(X, methods=["Raw", "SGS", "MSC"], sample_idx=0):
    X_sgs = apply_sgs(X)
    #X_snv = apply_snv(X)
    X_msc = apply_msc(X)
    spectra = {
        "Raw": X[sample_idx],
        "SGS": X_sgs[sample_idx],
        #"SNV": X_snv[sample_idx],
        "MSC": X_msc[sample_idx]
    }

    plt.figure(figsize=(12, 6))
    for name in methods:
        plt.plot(spectra[name], label=name)
    plt.title(f"Preprocessing Effects on Spectrum #{sample_idx}")
    plt.xlabel("Spectral Band Index")
    plt.ylabel("Intensity (A.U.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Optional: visualize preprocessing for one spectrum
plot_preprocessing(X_raw, sample_idx=3)

# ========================================
# 4. Choose Preprocessing for Modeling
# ========================================

# Uncomment one of the following:
#X_processed = apply_sgs(X_filtered)
# X_processed = apply_snv(X_raw)
# X_processed = apply_msc(X_raw)
X_processed_1 = X_filtered
#X_processed = apply_snv(X_processed_1)
X_processed = apply_msc(X_processed_1)
# =====================================
# Path to the .txt file
wl_filepath = open(r"D:\Python_Codes_N_Results_Folder\Results\Cubert_wavelength_data\wavelength_cubertdata.txt",)
wl_csv=wl_filepath.read()
wavelengths = np.array([float(val.strip()) for val in wl_csv.split(',') if val.strip() != ''])
#print(wavelengths)
wavelengths_M=wavelengths[:145]
#print(wavelengths_M)
#exit()
# ===========================
# (Assumes X_processed and y already loaded/defined)
# Make sure X_processed is (samples, 145), y is 0/1
X = X_processed.astype(np.float32)
y = y.astype(int)

# ===========================
# SMOTE for balancing classes
# ===========================
X_res, y_res = SMOTE().fit_resample(X, y)

# Expand dims for Conv1D
X_res = X_res[..., np.newaxis]

# ===========================
# Train/test split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
)

# ===========================
# Compute class weights (optional if using SMOTE)
# ===========================
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights))

# ===========================
# Build deeper 1D-CNN model
# ===========================
model = Sequential([
    Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(145,1)),
    BatchNormalization(),
    Conv1D(64, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv1D(128, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# ===========================
# Compile the model
# ===========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===========================
# Train with callbacks
# ===========================
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    class_weight=class_weights,  # optional if you want extra emphasis
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ===========================
# Evaluation
# ===========================
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['Healthy','Mildew']))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy','Mildew']).plot(
    cmap=plt.cm.Blues
)
plt.title('Confusion Matrix - Improved CNN')
plt.show()
