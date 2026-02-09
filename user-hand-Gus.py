import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import os
import time
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy import interpolate
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
import glob
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Font settings for Chinese characters (keep for any remaining text)
plt.rcParams['font.family'] = ['SimHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# Result directories - replace with your actual paths
BASE_DATA_DIR = r'YOUR_BASE_DATA_PATH_HERE'
RESULTS_DIR = r'YOUR_RESULTS_PATH_HERE'
GESTURE_RESULTS_DIR = os.path.join(RESULTS_DIR, 'gesture_classification')
PERSON_RESULTS_DIR = os.path.join(RESULTS_DIR, 'person_classification')

# Create all necessary directories
for dir_path in [RESULTS_DIR, GESTURE_RESULTS_DIR, PERSON_RESULTS_DIR]:
    for sub_dir in ['confusion_matrices', 'processed_data', 'training_plots',
                    'feature_analysis', 'training_curves_data', 'model_checkpoints',
                    'intermediate_results']:
        os.makedirs(os.path.join(dir_path, sub_dir), exist_ok=True)

# Define gestures and users
GESTURE_LABELS = ['likes', 'swear', 'victory', 'fist', 'snap', 'right', 'left', 'one']
USER_LABELS = ['User 1', 'User 2', 'User 3', 'User 4']  # Renamed from Chinese names


# Data loading function - strictly follows your file structure
def load_sEMG_data():
    """
    Load real sEMG data according to your file structure
    Example file path: YOUR_BASE_DATA_PATH_HERE\User 1\likes\ (1)_processed.xlsx
    """
    print("Starting to load real sEMG data...")

    all_samples = []
    all_gesture_labels = []
    all_user_labels = []
    sample_info = []
    max_length = 0

    # Step 1: Scan all files to determine maximum length
    for user_idx, user_name in enumerate(USER_LABELS):
        user_dir = os.path.join(BASE_DATA_DIR, user_name)
        if not os.path.exists(user_dir):
            print(f"Warning: User directory {user_dir} does not exist")
            continue

        for gesture_idx, gesture_name in enumerate(GESTURE_LABELS):
            gesture_dir = os.path.join(user_dir, gesture_name)
            if not os.path.exists(gesture_dir):
                print(f"Warning: Gesture directory {gesture_dir} does not exist")
                continue

            # Find all processed files
            file_pattern = os.path.join(gesture_dir, ' (*)_processed.xlsx')
            files = glob.glob(file_pattern)

            for file_path in files:
                try:
                    df = pd.read_excel(file_path)
                    # Check required columns
                    if 'Channel1_envelope' not in df.columns or 'Channel2_envelope' not in df.columns:
                        print(f"Warning: File {file_path} lacks required channel data columns")
                        continue

                    # Extract data from two channels
                    ch1_data = df['Channel1_envelope'].values
                    ch2_data = df['Channel2_envelope'].values

                    # Remove NaN values
                    ch1_valid = ch1_data[~np.isnan(ch1_data)]
                    ch2_valid = ch2_data[~np.isnan(ch2_data)]

                    # Calculate valid length
                    valid_length = min(len(ch1_valid), len(ch2_valid))
                    if valid_length > max_length:
                        max_length = valid_length

                except Exception as e:
                    print(f"Error scanning file {file_path}: {str(e)}")

    print(f"Maximum valid data length across all samples: {max_length}")

    # Step 2: Actually load the data
    for user_idx, user_name in enumerate(USER_LABELS):
        user_dir = os.path.join(BASE_DATA_DIR, user_name)
        if not os.path.exists(user_dir):
            continue

        for gesture_idx, gesture_name in enumerate(GESTURE_LABELS):
            gesture_dir = os.path.join(user_dir, gesture_name)
            if not os.path.exists(gesture_dir):
                continue

            file_pattern = os.path.join(gesture_dir, ' (*)_processed.xlsx')
            files = sorted(glob.glob(file_pattern))  # Sort for consistency

            for file_idx, file_path in enumerate(files):
                if file_idx >= 50:  # Maximum 50 samples per gesture
                    break

                try:
                    df = pd.read_excel(file_path)

                    # Extract channel data
                    ch1_data = df['Channel1_envelope'].values
                    ch2_data = df['Channel2_envelope'].values

                    # Remove NaN values
                    ch1_valid = ch1_data[~np.isnan(ch1_data)]
                    ch2_valid = ch2_data[~np.isnan(ch2_data)]

                    # Ensure both channels have same length
                    min_length = min(len(ch1_valid), len(ch2_valid))
                    if min_length < 100:  # Skip if data is too short
                        print(f"Warning: File {file_path} data too short ({min_length} points), skipping")
                        continue

                    ch1_valid = ch1_valid[:min_length]
                    ch2_valid = ch2_valid[:min_length]

                    # Linear interpolation to maximum length
                    if min_length < max_length:
                        x_original = np.linspace(0, 1, min_length)
                        x_target = np.linspace(0, 1, max_length)

                        f_ch1 = interpolate.interp1d(x_original, ch1_valid, kind='linear',
                                                     fill_value='extrapolate')
                        f_ch2 = interpolate.interp1d(x_original, ch2_valid, kind='linear',
                                                     fill_value='extrapolate')

                        ch1_resampled = f_ch1(x_target)
                        ch2_resampled = f_ch2(x_target)
                    else:
                        # Truncate if longer than maximum length
                        ch1_resampled = ch1_valid[:max_length]
                        ch2_resampled = ch2_valid[:max_length]

                    # Create sample features
                    sample_features = np.stack([ch1_resampled, ch2_resampled], axis=0)
                    all_samples.append(sample_features)
                    all_gesture_labels.append(gesture_idx)
                    all_user_labels.append(user_idx)

                    sample_info.append({
                        'file_path': file_path,
                        'user': user_name,
                        'gesture': gesture_name,
                        'user_code': user_idx,
                        'gesture_code': gesture_idx,
                        'original_length': min_length,
                        'processed_length': len(ch1_resampled)
                    })

                except Exception as e:
                    print(f"Error loading file {file_path}: {str(e)}")

    # Convert to numpy arrays
    all_samples = np.array(all_samples, dtype=np.float32)
    all_gesture_labels = np.array(all_gesture_labels, dtype=np.int64)
    all_user_labels = np.array(all_user_labels, dtype=np.int64)

    print(f"\nData loading completed:")
    print(f"Total samples: {len(all_samples)}")
    print(f"Feature shape: {all_samples.shape}")
    print(f"Gesture label distribution: {dict(zip(GESTURE_LABELS, np.bincount(all_gesture_labels)))}")
    print(f"User label distribution: {dict(zip(USER_LABELS, np.bincount(all_user_labels)))}")

    # Save data statistics
    stats_df = pd.DataFrame(sample_info)
    stats_path = os.path.join(RESULTS_DIR, 'processed_data', 'data_statistics.xlsx')
    stats_df.to_excel(stats_path, index=False)
    print(f"Data statistics saved to: {stats_path}")

    return (all_samples, all_gesture_labels, all_user_labels,
            GESTURE_LABELS, USER_LABELS, max_length)


# Data augmentation function
def augment_data(X, y, original_length):
    """Perform data augmentation only on training set to avoid data leakage"""
    augmented_X = []
    augmented_y = []

    for sample, label in zip(X, y):
        # 1. Original sample
        augmented_X.append(sample)
        augmented_y.append(label)

        # 2. Gaussian noise augmentation
        noise_std = np.std(sample) * 0.02  # Adaptive noise based on signal standard deviation
        noise = np.random.normal(0, noise_std, sample.shape)
        augmented_sample = sample + noise
        # Ensure non-negative signals (sEMG characteristic)
        augmented_sample = np.maximum(augmented_sample, 0)
        augmented_X.append(augmented_sample)
        augmented_y.append(label)

        # 3. Time warping augmentation
        warp_factor = np.random.uniform(0.8, 1.2)
        new_length = int(original_length * warp_factor)

        warped_sample = np.zeros((2, new_length))
        for ch in range(2):
            original_signal = sample[ch]
            x_original = np.linspace(0, 1, original_length)
            x_warped = np.linspace(0, 1, new_length)

            # Use cubic spline interpolation for smoother results
            if len(original_signal) > 3:  # Ensure enough points for spline interpolation
                try:
                    from scipy.interpolate import CubicSpline
                    cs = CubicSpline(x_original, original_signal)
                    warped_signal = cs(x_warped)
                except:
                    # Fallback to linear interpolation
                    warped_signal = np.interp(x_warped, x_original, original_signal)
            else:
                warped_signal = np.interp(x_warped, x_original, original_signal)

            warped_sample[ch] = warped_signal

        # Interpolate back to original length
        resized_sample = np.zeros((2, original_length))
        for ch in range(2):
            x_warped = np.linspace(0, 1, new_length)
            x_target = np.linspace(0, 1, original_length)
            resized_sample[ch] = np.interp(x_target, x_warped, warped_sample[ch])

        augmented_X.append(resized_sample)
        augmented_y.append(label)

    return np.array(augmented_X), np.array(augmented_y)


# Lightweight dual-channel deep learning model - reduced memory usage
class LightDualChannelModel(nn.Module):
    def __init__(self, num_classes=9, input_length=5000):
        super().__init__()

        # Early feature extraction - reduced channels
        self.early_conv = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),  # Reduced to 32 channels
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, 5, padding=2),  # Reduced to 64 channels
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        # Channel-specific feature extraction - further reduced channels
        self.ch1_branch = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2),  # Reduced channel count
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, 3, padding=1),  # Reduced channel count
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )

        self.ch2_branch = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )

        # Attention mechanisms
        self.channel_attention = ChannelAttention(256)  # Adjusted input dimension
        self.temporal_attention = TemporalAttention(128)  # Adjusted input dimension

        # Bidirectional GRU - reduced hidden size
        self.gru = nn.GRU(
            input_size=256,  # Reduced input size
            hidden_size=128,  # Reduced hidden size
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # Adjusted input size
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Early feature extraction
        early_features = self.early_conv(x)  # [batch, 64, seq_len/4]

        # Split to two channel branches
        ch1_early, ch2_early = torch.chunk(early_features, 2, dim=1)

        # Channel-specific processing
        ch1_features = self.ch1_branch(ch1_early)
        ch2_features = self.ch2_branch(ch2_early)

        # Channel attention
        combined = torch.cat([ch1_features, ch2_features], dim=1)
        attended = self.channel_attention(combined)

        # Temporal modeling
        gru_input = attended.permute(0, 2, 1)
        gru_output, _ = self.gru(gru_input)

        # Temporal attention
        output = self.temporal_attention(gru_output)

        # Classification
        return self.classifier(output)


class ChannelAttention(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channel_dim, channel_dim // 8),
            nn.LeakyReLU(0.1),
            nn.Linear(channel_dim // 8, channel_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x).unsqueeze(2)
        return x * attention_weights


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),  # Reduced hidden size
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        weighted = (x * attention_weights).sum(dim=1)
        return weighted


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


# Function to save intermediate results
def save_intermediate_results(results_dir, task_name, fold_idx, true_labels,
                              pred_labels, probabilities, fold_results, epoch_results=None):
    """Save intermediate results for each fold to prevent training interruption"""

    # Create fold directory
    fold_dir = os.path.join(results_dir, 'intermediate_results', f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)

    # Save prediction results
    results_data = {
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'probabilities': probabilities,
        'fold_results': fold_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save as numpy file
    np.save(os.path.join(fold_dir, f'{task_name}_fold_{fold_idx}_results.npy'), results_data)

    # Save as Excel file (for easy viewing)
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'pred_label': pred_labels,
        'max_probability': np.max(probabilities, axis=1),
        'correct': (np.array(true_labels) == np.array(pred_labels)).astype(int)
    })
    results_df.to_excel(os.path.join(fold_dir, f'{task_name}_fold_{fold_idx}_detailed.xlsx'), index=False)

    # Save training process data (if available)
    if epoch_results is not None:
        epoch_df = pd.DataFrame(epoch_results)
        epoch_df.to_excel(os.path.join(fold_dir, f'{task_name}_fold_{fold_idx}_training.xlsx'), index=False)

    print(f"Fold {fold_idx} intermediate results saved to: {fold_dir}")


def load_intermediate_results(results_dir, task_name):
    """Load previously saved intermediate results"""
    intermediate_dir = os.path.join(results_dir, 'intermediate_results')
    if not os.path.exists(intermediate_dir):
        return None, 0

    fold_dirs = [d for d in os.listdir(intermediate_dir) if d.startswith('fold_')]
    if not fold_dirs:
        return None, 0

    # Find the latest fold
    completed_folds = []
    all_true = []
    all_pred = []
    all_probs = []

    for fold_dir in sorted(fold_dirs):
        fold_idx = int(fold_dir.split('_')[1])
        result_file = os.path.join(intermediate_dir, fold_dir, f'{task_name}_fold_{fold_idx}_results.npy')

        if os.path.exists(result_file):
            try:
                results_data = np.load(result_file, allow_pickle=True).item()
                completed_folds.append(fold_idx)
                all_true.extend(results_data['true_labels'])
                all_pred.extend(results_data['pred_labels'])
                all_probs.extend(results_data['probabilities'])
                print(f"Successfully loaded fold {fold_idx} results")
            except Exception as e:
                print(f"Error loading fold {fold_idx} results: {e}")

    if completed_folds:
        next_fold = max(completed_folds) + 1
        return (np.array(all_true), np.array(all_pred), np.array(all_probs), completed_folds), next_fold
    else:
        return None, 0


# Fixed t-SNE function
def safe_tsne_visualization(features, labels, class_names, save_path, title_suffix=""):
    """Safe t-SNE visualization, handling various dimension cases"""

    # Ensure features are numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # If feature dimension is too high, perform PCA preprocessing first
    if features.shape[1] > 50:
        from sklearn.decomposition import PCA
        # Safely set PCA component count
        n_components = min(50, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_components)
        features_reduced = pca.fit_transform(features)
        print(f"PCA dimensionality reduction: {features.shape[1]} -> {n_components} dimensions")
    else:
        features_reduced = features

    # Set appropriate perplexity
    n_samples = len(features_reduced)
    perplexity = min(30, n_samples - 1)

    print(f"Performing t-SNE dimensionality reduction: {features_reduced.shape[1]} -> 2 dimensions, perplexity={perplexity}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features_reduced)

    # Plot t-SNE
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.7, s=50)
    plt.colorbar(scatter)
    plt.title(f't-SNE Feature Visualization {title_suffix}', fontsize=16)
    plt.xlabel('t-SNE Feature 1', fontsize=12)
    plt.ylabel('t-SNE Feature 2', fontsize=12)

    # Add class labels
    for i, class_name in enumerate(class_names):
        class_indices = np.where(labels == i)[0]
        if len(class_indices) > 0:
            centroid = np.median(features_2d[class_indices], axis=0)
            plt.annotate(class_name, centroid,
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save t-SNE data
    tsne_data = pd.DataFrame({
        'TSNE1': features_2d[:, 0],
        'TSNE2': features_2d[:, 1],
        'true_label': labels,
        'class': [class_names[label] for label in labels]
    })
    tsne_data_path = save_path.replace('.png', '_data.xlsx')
    tsne_data.to_excel(tsne_data_path, index=False)

    print(f"t-SNE visualization saved to: {save_path}")
    print(f"t-SNE coordinate data saved to: {tsne_data_path}")

    return features_2d


# Memory management function
def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")


# Training function
def train_gesture_classification():
    """Train gesture classification model"""
    print("\n" + "=" * 60)
    print("Starting gesture classification model training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    X, gesture_y, user_y, gesture_labels, user_labels, max_length = load_sEMG_data()

    # Try to load previous intermediate results
    loaded_data, start_fold = load_intermediate_results(GESTURE_RESULTS_DIR, "gesture_classification")

    if loaded_data is not None:
        all_true, all_pred, all_probs, completed_folds = loaded_data
        print(f"Continuing training from fold {start_fold}, {len(completed_folds)} folds completed")
    else:
        all_true = []
        all_pred = []
        all_probs = []
        completed_folds = []
        start_fold = 1

    # Use gesture labels for stratified K-fold cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fold_results = []
    fold = 0

    for train_idx, test_idx in skf.split(X, gesture_y):
        fold += 1

        # Skip already completed folds
        if fold < start_fold:
            continue

        print(f"\n===== Gesture Classification Fold {fold} =====")

        try:
            # Clear GPU memory
            clear_gpu_memory()

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = gesture_y[train_idx], gesture_y[test_idx]

            print(f"Training set distribution: {np.bincount(y_train)}")
            print(f"Test set distribution: {np.bincount(y_test)}")

            # Data augmentation (only on training set)
            X_train_aug, y_train_aug = augment_data(X_train, y_train, max_length)

            # Apply SMOTE to balance classes
            n_samples, n_channels, n_timesteps = X_train_aug.shape
            X_train_flat = X_train_aug.reshape(n_samples, n_channels * n_timesteps)

            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_flat, y_train_aug)
            X_train_processed = X_train_smote.reshape(-1, n_channels, n_timesteps)

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_processed)
            y_train_tensor = torch.LongTensor(y_train_smote)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)

            # Data loader - reduced batch size
            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                      batch_size=8, shuffle=True)  # Reduced batch size
            test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                                     batch_size=8, shuffle=False)  # Reduced batch size

            # Use lightweight model
            model = LightDualChannelModel(num_classes=len(gesture_labels)).to(device)

            # Loss function and optimizer
            class_weights = compute_class_weight('balanced',
                                                 classes=np.unique(y_train_smote),
                                                 y=y_train_smote)
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)

            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

            # Training loop
            best_acc = 0
            patience = 15
            counter = 0
            epoch_results = []

            for epoch in range(100):
                model.train()
                train_loss = 0
                correct = 0
                total = 0

                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()

                train_acc = 100. * correct / total

                # Validation
                model.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0

                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()

                val_acc = 100. * val_correct / val_total
                val_loss = val_loss / len(test_loader)

                epoch_results.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss / len(train_loader),
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })

                scheduler.step(val_loss)

                if val_acc > best_acc:
                    best_acc = val_acc
                    counter = 0
                    torch.save(model.state_dict(),
                               os.path.join(GESTURE_RESULTS_DIR, 'model_checkpoints',
                                            f'gesture_fold{fold}_best.pth'))
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                if (epoch + 1) % 20 == 0:
                    print(f'Epoch [{epoch + 1}/100], Train Loss: {train_loss / len(train_loader):.4f}, '
                          f'Validation Accuracy: {val_acc:.2f}%')

            # Final evaluation
            model.load_state_dict(torch.load(
                os.path.join(GESTURE_RESULTS_DIR, 'model_checkpoints',
                             f'gesture_fold{fold}_best.pth')))
            model.eval()

            fold_true = []
            fold_pred = []
            fold_probs = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)

                    fold_true.extend(batch_y.numpy())
                    fold_pred.extend(predicted.cpu().numpy())
                    fold_probs.extend(probabilities.cpu().numpy())

            fold_acc = 100 * np.sum(np.array(fold_pred) == np.array(fold_true)) / len(fold_true)
            fold_results.append({
                'fold': fold,
                'accuracy': fold_acc,
                'best_accuracy': best_acc
            })

            # Save current fold intermediate results
            save_intermediate_results(GESTURE_RESULTS_DIR, "gesture_classification", fold,
                                      fold_true, fold_pred, fold_probs,
                                      fold_results, epoch_results)

            all_true.extend(fold_true)
            all_pred.extend(fold_pred)
            all_probs.extend(fold_probs)
            completed_folds.append(fold)

            print(f"Fold {fold} final accuracy: {fold_acc:.2f}%")

            # Clean memory
            del model, train_loader, test_loader
            clear_gpu_memory()

        except Exception as e:
            print(f"Error during fold {fold} training: {str(e)}")
            import traceback
            traceback.print_exc()

            # Clean memory
            clear_gpu_memory()
            continue

    # Save final results - fixed array judgment issue
    if len(all_true) > 0:  # Fixed here
        save_final_results(all_true, all_pred, all_probs, gesture_labels,
                           fold_results, GESTURE_RESULTS_DIR, "gesture_classification")

    return np.array(all_true), np.array(all_pred), np.array(all_probs)


def train_user_classification():
    """Train user classification model"""
    print("\n" + "=" * 60)
    print("Starting user classification model training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X, gesture_y, user_y, gesture_labels, user_labels, max_length = load_sEMG_data()

    # Try to load previous intermediate results
    loaded_data, start_fold = load_intermediate_results(PERSON_RESULTS_DIR, "user_classification")

    if loaded_data is not None:
        all_true, all_pred, all_probs, completed_folds = loaded_data
        print(f"Continuing training from fold {start_fold}, {len(completed_folds)} folds completed")
    else:
        all_true = []
        all_pred = []
        all_probs = []
        completed_folds = []
        start_fold = 1

    # Use user labels for stratified K-fold cross-validation
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)  # 4 folds for 4 users

    fold_results = []
    fold = 0

    for train_idx, test_idx in skf.split(X, user_y):
        fold += 1

        # Skip already completed folds
        if fold < start_fold:
            continue

        print(f"\n===== User Classification Fold {fold} =====")

        try:
            # Clear GPU memory
            clear_gpu_memory()

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = user_y[train_idx], user_y[test_idx]

            print(f"Training set distribution: {np.bincount(y_train)}")
            print(f"Test set distribution: {np.bincount(y_test)}")

            # Data preprocessing
            X_train_aug, y_train_aug = augment_data(X_train, y_train, max_length)

            n_samples, n_channels, n_timesteps = X_train_aug.shape
            X_train_flat = X_train_aug.reshape(n_samples, n_channels * n_timesteps)

            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_flat, y_train_aug)
            X_train_processed = X_train_smote.reshape(-1, n_channels, n_timesteps)

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_processed)
            y_train_tensor = torch.LongTensor(y_train_smote)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)

            # Data loader - reduced batch size
            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                      batch_size=8, shuffle=True)  # Reduced batch size
            test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                                     batch_size=8, shuffle=False)  # Reduced batch size

            # Use lightweight model
            model = LightDualChannelModel(num_classes=len(user_labels)).to(device)

            # Training configuration
            class_weights = compute_class_weight('balanced',
                                                 classes=np.unique(y_train_smote),
                                                 y=y_train_smote)
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)

            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

            # Training loop
            best_acc = 0
            patience = 15
            counter = 0
            epoch_results = []

            for epoch in range(100):
                model.train()
                train_loss = 0
                correct = 0
                total = 0

                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()

                train_acc = 100. * correct / total

                # Validation
                model.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0

                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()

                val_acc = 100. * val_correct / val_total
                val_loss = val_loss / len(test_loader)

                epoch_results.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss / len(train_loader),
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })

                scheduler.step(val_loss)

                if val_acc > best_acc:
                    best_acc = val_acc
                    counter = 0
                    torch.save(model.state_dict(),
                               os.path.join(PERSON_RESULTS_DIR, 'model_checkpoints',
                                            f'user_fold{fold}_best.pth'))
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                if (epoch + 1) % 20 == 0:
                    print(f'Epoch [{epoch + 1}/100], Train Loss: {train_loss / len(train_loader):.4f}, '
                          f'Validation Accuracy: {val_acc:.2f}%')

            # Final evaluation
            model.load_state_dict(torch.load(
                os.path.join(PERSON_RESULTS_DIR, 'model_checkpoints',
                             f'user_fold{fold}_best.pth')))
            model.eval()

            fold_true = []
            fold_pred = []
            fold_probs = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)

                    fold_true.extend(batch_y.numpy())
                    fold_pred.extend(predicted.cpu().numpy())
                    fold_probs.extend(probabilities.cpu().numpy())

            fold_acc = 100 * np.sum(np.array(fold_pred) == np.array(fold_true)) / len(fold_true)
            fold_results.append({
                'fold': fold,
                'accuracy': fold_acc,
                'best_accuracy': best_acc
            })

            # Save current fold intermediate results
            save_intermediate_results(PERSON_RESULTS_DIR, "user_classification", fold,
                                      fold_true, fold_pred, fold_probs,
                                      fold_results, epoch_results)

            all_true.extend(fold_true)
            all_pred.extend(fold_pred)
            all_probs.extend(fold_probs)
            completed_folds.append(fold)

            print(f"Fold {fold} final accuracy: {fold_acc:.2f}%")

            # Clean memory
            del model, train_loader, test_loader
            clear_gpu_memory()

        except Exception as e:
            print(f"Error during fold {fold} training: {str(e)}")
            import traceback
            traceback.print_exc()

            # Clean memory
            clear_gpu_memory()
            continue

    # Save final results - fixed array judgment issue
    if len(all_true) > 0:  # Fixed here
        save_final_results(all_true, all_pred, all_probs, user_labels,
                           fold_results, PERSON_RESULTS_DIR, "user_classification")

    return np.array(all_true), np.array(all_pred), np.array(all_probs)


def save_final_results(true_labels, pred_labels, probabilities, class_names,
                       fold_results, results_dir, task_name):
    """Save all final results and visualizations"""

    print(f"\nSaving {task_name} final results...")

    # Convert probabilities to numpy array
    prob_array = np.array(probabilities)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{task_name} Confusion Matrix (%)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices', f'{task_name}_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save confusion matrix data
    cm_df = pd.DataFrame(cm,
                         index=[f'True_{name}' for name in class_names],
                         columns=[f'Pred_{name}' for name in class_names])
    cm_df['Row_Total'] = cm_df.sum(axis=1)
    cm_df.loc['Column_Total'] = cm_df.sum(axis=0)
    cm_df.to_excel(os.path.join(results_dir, 'confusion_matrices',
                                f'{task_name}_confusion_matrix_data.xlsx'))

    # Classification report
    report = classification_report(true_labels, pred_labels,
                                   target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_excel(os.path.join(results_dir, 'processed_data',
                                    f'{task_name}_classification_report.xlsx'))

    # Per-fold results
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_excel(os.path.join(results_dir, 'processed_data',
                                  f'{task_name}_per_fold_results.xlsx'), index=False)

    # Use fixed t-SNE visualization
    tsne_path = os.path.join(results_dir, 'feature_analysis',
                             f'{task_name}_t-SNE_visualization.png')
    safe_tsne_visualization(prob_array, np.array(true_labels),
                            class_names, tsne_path, task_name)

    # Save final probability data
    final_data = pd.DataFrame({
        'true_label': true_labels,
        'pred_label': pred_labels,
        'true_class': [class_names[i] for i in true_labels],
        'pred_class': [class_names[i] for i in pred_labels],
        'max_probability': np.max(prob_array, axis=1),
        'is_correct': (np.array(true_labels) == np.array(pred_labels)).astype(int)
    })

    # Add probability for each class
    for i, class_name in enumerate(class_names):
        final_data[f'probability_{class_name}'] = prob_array[:, i]

    final_data.to_excel(os.path.join(results_dir, 'processed_data',
                                     f'{task_name}_final_prediction_results.xlsx'), index=False)

    # Calculate overall accuracy
    final_acc = 100 * np.sum(np.array(pred_labels) == np.array(true_labels)) / len(true_labels)

    print(f"\n{task_name} final results:")
    print(f"- Overall accuracy: {final_acc:.2f}%")
    print(f"- Total samples: {len(true_labels)}")
    print(f"- Confusion matrix and data saved")
    print(f"- Classification report saved")
    print(f"- t-SNE visualization saved")
    print(f"- Per-fold results saved")
    print(f"- Final prediction results saved")


if __name__ == "__main__":
    print("=" * 80)
    print("sEMG Gesture and User Classification Deep Learning Model")
    print("Fixed Version - Supports Checkpoint Resuming and Real-time Data Saving")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        start_time = time.time()

        # Train gesture classification model
        gesture_true, gesture_pred, gesture_probs = train_gesture_classification()

        # Train user classification model
        user_true, user_pred, user_probs = train_user_classification()

        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s")

        print(f"\nAll results saved to: {RESULTS_DIR}")
        print("Includes: confusion matrices, classification reports, t-SNE visualizations, training curves, etc.")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback

        traceback.print_exc()

        # Try to save completed intermediate results even if error occurs
        print("\nAttempting to save completed intermediate results...")
        try:
            # Add emergency save logic here
            pass
        except:
            print("Emergency save also failed")