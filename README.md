# Hey Lora - Wake Word Detection

A custom wake word detection system that recognizes the phrase **"Hey Lora"** from audio input. The model is trained from scratch using a CNN-BiLSTM-Attention architecture on ~18,000 audio samples, achieving **0.977 F1-score** on the held-out test set.

## Results

| Metric    | Test Set |
|-----------|----------|
| Accuracy  | 99.28%   |
| F1 Score  | 0.9772   |
| Precision | 0.9789   |
| Recall    | 0.9754   |

**Confusion Matrix (Test Set)**:
|                  | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| Actual Negative  | 1524               | 6                  |
| Actual Positive  | 7                  | 278                |

Only 13 misclassifications out of 1,815 test samples (6 false positives, 7 false negatives).

## Dataset

- **Total samples**: 18,147 audio clips
- **Positive class** (wake word "Hey Lora"): 2,849 samples (15.7%)
- **Negative class** (non-wake word audio): 15,298 samples (84.3%)
- **Source**: Mozilla Common Voice dataset with custom wake word recordings
- **Audio format**: 1-second clips, resampled to 16 kHz mono

The dataset is split using stratified sampling to preserve class proportions:

| Split      | Samples | Positive | Negative |
|------------|---------|----------|----------|
| Train      | 13,610  | 2,137    | 11,473   |
| Validation | 2,722   | 427      | 2,295    |
| Test       | 1,815   | 285      | 1,530    |

## Architecture

The model (`HeyLoraNet`) combines convolutional feature extraction with sequential modeling:

```
Audio Waveform
    |
Mel Spectrogram (64 mels, 63 time frames)
    |
CNN Feature Extractor
    ├── Stem Conv2d (1 → 32 channels)
    ├── ResidualBlock + SE Attention (32 → 64, stride=2)
    ├── TemporalBlock with (3x7) kernel (64 → 128, stride=2)
    └── ResidualBlock + SE Attention (128 → 128)
    |
AdaptiveAvgPool2d → (128, 8, 16)
    |
Reshape to sequence (16 timesteps, 1024 features)
    |
Learnable Positional Encoding
    |
BiLSTM (2 layers, hidden=128) + LayerNorm
    |
Multi-Head Self-Attention (4 heads) + Residual + LayerNorm
    |
Mean Pooling ⊕ Max Pooling (concatenated)
    |
Classifier (FC → ReLU → Dropout → FC → 2 classes)
```

### Key Components

- **Squeeze-and-Excitation (SE) Blocks**: Channel attention mechanism that lets the network learn to weight feature channels by importance.
- **Temporal Block**: Uses an asymmetric (3x7) convolution kernel to capture temporal patterns in the spectrogram, which are critical for keyword spotting.
- **Residual Connections**: Skip connections around each convolutional block to ease gradient flow.
- **BiLSTM**: Bidirectional LSTM captures both past and future context in the audio sequence.
- **Multi-Head Self-Attention**: Allows the model to attend to the most relevant parts of the audio, refining the LSTM output.
- **Dual Pooling**: Concatenation of mean and max pooling captures both average and peak activation patterns.

**Total parameters**: ~2.46M

## Audio Preprocessing

All audio passes through a preprocessing pipeline implemented as a custom `collate_fn`:

1. **Resampling**: All audio is resampled to 16 kHz
2. **Mono conversion**: Multi-channel audio is averaged to mono
3. **Amplitude normalization**: Peak-normalized to 0.95
4. **Pad/Trim**: Waveforms are adjusted to exactly 1 second (16,000 samples). During training, random cropping/padding is used; during evaluation, center cropping/padding is used
5. **Mel Spectrogram**: 64 mel bands, 1024-point FFT, hop length 256, frequency range 50-8000 Hz
6. **Log scaling**: AmplitudeToDB with 80 dB dynamic range
7. **Instance normalization**: Per-sample zero-mean unit-variance normalization

Output tensor shape: `(batch, 1, 64, 63)` — 64 mel bands, 63 time frames.

## Data Augmentation

A class-aware augmentation strategy is applied during training to handle the class imbalance and improve generalization. The augmentation intensity varies by class: the minority class (wake word) receives more aggressive augmentation.

### Waveform-Level Augmentation (3 tiers)

| Tier   | Transforms | Wake Word Prob | Non-Wake Prob |
|--------|-----------|----------------|---------------|
| Light  | Gain, Gaussian noise, time shift | 25% | 65% |
| Medium | + Pitch shift, time stretch, LP/HP filter | 40% | 25% |
| Strong | + Band-pass, clipping/tanh distortion, room simulation, MP3 compression | 35% | 10% |

Waveform augmentation is applied with 75% probability per sample.

### Spectrogram-Level Augmentation (SpecAugment)

- **Frequency masking**: 2 masks, up to 8 mel bins each (~12% of spectrum)
- **Time masking**: 2 masks, up to 12 frames each (~20% of duration)
- Applied with 50% probability per batch

## Training Procedure

### Loss Function

**Focal Loss** with gamma=2.5 and class-dependent alpha weights. Focal Loss down-weights easy examples and focuses the training signal on hard-to-classify samples, which is particularly effective for imbalanced datasets.

Alpha weights are derived from inverse class frequencies with a correction factor (k=1.3):
- Class 0 (non-wake): 0.408
- Class 1 (wake word): 0.771

### Optimizer and Scheduler

- **Optimizer**: AdamW (lr=1.51e-3, weight_decay=0.01)
- **Learning rate**: Selected via LR Range Test (sweep from 1e-7 to 1e-1, pick the steepest descent point)
- **Scheduler**: OneCycleLR with cosine annealing, 25% warmup phase
- **Gradient clipping**: Max norm = 1.0

### Early Stopping

- Monitored metric: **Validation F1 Score**
- Patience: 40 epochs
- Minimum delta: 0.0005
- Best model checkpoint is saved automatically

### Weight Initialization

- **Conv2d**: Kaiming Normal (fan_out, ReLU)
- **BatchNorm2d**: weight=1, bias=0
- **Linear**: Xavier Normal
- **LSTM**: Xavier Normal for input weights, Orthogonal for hidden weights

## Tech Stack

- **PyTorch** + **torchaudio** — model, training, audio I/O and transforms
- **audiomentations** — waveform augmentation pipeline
- **scikit-learn** — stratified splitting, metrics (F1, precision, recall, confusion matrix)
- **torchinfo** — model summary and parameter counting

## Project Structure

```
wake_word/
├── main_f1.ipynb     # Full training pipeline (data → model → evaluation)
├── .gitignore
└── README.md
```