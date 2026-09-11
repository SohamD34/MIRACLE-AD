"""Model backbones and MIL pooling networks used by MIRACLE-AD.

The paper focuses on the two Gammatone backbones, while the additional
backbones from the research workspace remain available for ablations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np
import math
import copy

class BackboneType(enum.Enum):
    """Enum for available backbone architectures"""
    KURDISH_CNN = "kurdish_cnn"
    GAMMA_ERB_CNN = "gamma_erb_cnn"
    GAMMA_GM_CNN = "gamma_gm_cnn"
    SOUNDNET8 = "soundnet8"
    SOUNDNET5 = "soundnet5"
    M3 = 'm3'
    M5 = 'm5'
    M11 = 'm11'
    M18 = 'm18'
    RAW_AUDIO_CNN = "raw_audio_cnn"
    WAVENET = "wavenet"
    RAW_AUDIO_LSTM = "raw_audio_lstm"
    WAV2VEC2 = "wav2vec2"


class NetworkType(enum.Enum):
    """Enum for available classification networks"""
    ABMIL = "abmil"
    TRANSFORMER_ABMIL = "transformer_abmil"
    GATED_ABMIL = "gated_abmil"
    BILSTM = "bilstm"
    UNILSTM = "unilstm"
    # Backward-compatible aliases used by early scripts/checkpoints.
    BILSTM_DUAL_HEAD = "bilstm_dual_head"
    UNILSTM_DUAL_HEAD = "unilstm_dual_head"


# ==================== Gammatone Filter Layers ====================

class GammaConv1dGreenwoodERB(nn.Module):
    def __init__(self, num_filters=64, kernel_size=401, sr=24000,
                 low_freq=0, high_freq=8000, order=4, freeze=False):
        super(GammaConv1dGreenwoodERB, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.sr = sr
        self.order = order

        # Generate ERB-spaced center frequencies
        self.center_freqs = self._erb_space(low_freq, high_freq, num_filters)

        # Create filterbank weights [num_filters, 1, kernel_size]
        filters = []
        for cf in self.center_freqs:
            gt = self._gammatone_ir(cf, kernel_size, sr, order)
            filters.append(gt)
        filters = np.stack(filters)  # shape: [num_filters, kernel_size]
        filters = torch.tensor(filters, dtype=torch.float32).unsqueeze(1)  # [num_filters, 1, kernel_size]

        # Conv1d with fixed weights
        self.conv = nn.Conv1d(in_channels=1, out_channels=num_filters,
                              kernel_size=kernel_size, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(filters)
        if freeze:
            self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

    def _erb_space(self, low_freq, high_freq, num):
        """ERB scale spacing (Glasberg & Moore)"""
        EarQ = 9.26449
        minBW = 24.7
        cf = -(EarQ * minBW) + np.exp(
            np.linspace(1, num, num) *
            (-np.log(high_freq + EarQ * minBW) + np.log(low_freq + EarQ * minBW)) / num
        ) * (high_freq + EarQ * minBW)
        return cf

    def _gammatone_ir(self, cf, length, sr, order):
        """Generate gammatone filter impulse response."""
        t = np.arange(0, length) / sr
        erb = 24.7 + 0.108 * cf
        b = 1.019 * 2 * np.pi * erb
        a = t**(order - 1) * np.exp(-b * t) * np.cos(2 * np.pi * cf * t)
        # Normalize
        a /= np.max(np.abs(a))
        return a


class GammaConv1dGlasbergMoore(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, sample_rate,
                 order=4, min_freq=50, max_freq=None, freeze=False, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        if max_freq is None:
            max_freq = sample_rate / 2

        # Generate gammatone filterbank
        weight = self._make_gammatone_filterbank(
            out_channels, kernel_size, sample_rate, order, min_freq, max_freq
        )

        with torch.no_grad():
            self.weight.copy_(weight.repeat(in_channels, 1, 1))  # repeat for in_channels
            if self.bias is not None:
                self.bias.zero_()

        # Optionally freeze weights
        if freeze:
            self.weight.requires_grad = False

    def _make_gammatone_filterbank(self, n_filters, kernel_size, fs, order, min_freq, max_freq):
        freqs = np.geomspace(min_freq, max_freq, n_filters)  # log-spaced
        filters = []
        for fc in freqs:
            filters.append(self._gammatone_ir(fc, fs, kernel_size, order))
        filters = np.stack(filters)
        filters = filters / np.linalg.norm(filters, axis=1, keepdims=True)  # normalize
        return torch.tensor(filters, dtype=torch.float32).unsqueeze(1)  # (out, 1, kernel)

    def _gammatone_ir(self, fc, fs, length, order):
        t = np.arange(0, length) / fs
        erb = 24.7 + 0.108 * fc
        b = 1.019 * 2 * np.pi * erb
        gain = ((2 * np.pi * erb) ** order) / math.factorial(order - 1)
        g = gain * (t ** (order - 1)) * np.exp(-b * t) * np.cos(2 * np.pi * fc * t)
        return g.astype(np.float32)


# ==================== CNN Backbones ====================

class GammaERBCNNBackbone(nn.Module):
    """GammaERB CNN Backbone - outputs 1024-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 1024

        self.conv1 = GammaConv1dGreenwoodERB(
            sr=sample_rate,
            low_freq=50,
            high_freq=min(8000, sample_rate / 2),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, stride=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=8)

        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(), self.pool1,
            self.conv2,
            nn.ReLU(), self.pool2,
            self.conv3,
            nn.ReLU(),
            self.conv4,
            nn.ReLU(),
            self.adaptive_pool
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 128, 8] -> [B, 1024]
        x = self.features(x)
        return x.flatten(1)


class GammaGMCNNBackbone(nn.Module):
    """GammaGM CNN Backbone - outputs 1024-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 1024

        self.conv1 = GammaConv1dGlasbergMoore(
            in_channels=1,
            out_channels=64,
            kernel_size=64,
            sample_rate=sample_rate,
            min_freq=50,
            max_freq=min(8000, sample_rate / 2),
            freeze=False
        )
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=32, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=8)

        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(), self.pool1,
            self.conv2,
            nn.ReLU(), self.pool2,
            self.conv3,
            nn.ReLU(),
            self.conv4,
            nn.ReLU(),
            self.adaptive_pool
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 128, 8] -> [B, 1024]
        x = self.features(x)
        return x.flatten(1)


class SoundNet8Backbone(nn.Module):
    """SoundNet8 Backbone - outputs 1401-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 1401

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=1, padding=0),
            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=1, padding=0),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=8),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1, padding=0),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(1024, 1401, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 1401, 1] -> [B, 1401]
        x = self.features(x)
        return x.flatten(1)


class SoundNet5Backbone(nn.Module):
    """SoundNet5 Backbone - outputs 1401-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 1401

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(256, 1401, kernel_size=16, stride=12, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 1401, 1] -> [B, 1401]
        x = self.features(x)
        return x.flatten(1)


class KurdishCNNBackbone(nn.Module):
    """Kurdish CNN Backbone - outputs 100-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 100

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 100, 1] -> [B, 100]
        x = self.features(x)
        return x.flatten(1)


class M3Backbone(nn.Module):
    """M3 Backbone - outputs 256-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 256

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=80, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 256, 1] -> [B, 256]
        x = self.features(x)
        return x.flatten(1)


class M5Backbone(nn.Module):
    """M5 Backbone - outputs 512-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 512

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=80, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 512, 1] -> [B, 512]
        x = self.features(x)
        return x.flatten(1)


class M11Backbone(nn.Module):
    """M11 Backbone - outputs 512-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 512

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=80, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 512, 1] -> [B, 512]
        x = self.features(x)
        return x.flatten(1)


class M18Backbone(nn.Module):
    """M18 Backbone - outputs 512-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 512

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=80, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 512, 1] -> [B, 512]
        x = self.features(x)
        return x.flatten(1)


class RawAudioCNNBackbone(nn.Module):
    """Raw Audio CNN Backbone - outputs 256-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 256

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=40, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=20, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B, 1, L] -> [B, 256, 1] -> [B, 256]
        x = self.features(x)
        return x.flatten(1)


class WaveNetBackbone(nn.Module):
    """WaveNet Backbone - outputs 64-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 64

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1 if i == 0 else 64, 64, kernel_size=3, dilation=2**i, padding=2**i),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1 + 0.1*i)
            ) for i in range(8)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, 1, L]
        residual = None
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if residual is not None:
                x = x + residual
            residual = x

        x = self.pool(x)  # [B, 64, 1]
        return x.flatten(1)  # [B, 64]


class RawAudioLSTMBackbone(nn.Module):
    """Raw Audio LSTM Backbone - outputs 256-dim features"""
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.feature_dim = 256

        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=40, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.lstm = nn.LSTM(input_size=128, hidden_size=128,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)

        self.attention = nn.Linear(256, 1)

    def forward(self, x):
        # x: [B, 1, L]
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1)  # [B, T, 128]
        x, _ = self.lstm(x)  # [B, T, 256]

        # Attention pooling
        attn_weights = torch.softmax(self.attention(x), dim=1)  # [B, T, 1]
        context = torch.sum(x * attn_weights, dim=1)  # [B, 256]

        return context


class Wav2Vec2Backbone(nn.Module):
    """Wav2Vec2 Backbone - outputs 768-dim features (configurable model)"""
    def __init__(self, model_name="facebook/wav2vec2-base-960h", sample_rate=24000, cache_dir=None):
        super().__init__()

        try:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "Wav2Vec2 support is optional. Install it with "
                "`pip install -e '.[wav2vec]'`."
            ) from exc

        self.sample_rate = sample_rate
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, cache_dir=cache_dir)
        self.feature_dim = self.wav2vec2.config.hidden_size

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            return_attention_mask=True,
            cache_dir=cache_dir
        )
        self.model_sample_rate = int(self.feature_extractor.sampling_rate)

    def forward(self, x):
        # x: [B, 1, L] or [B, L]
        if x.dim() == 3:
            x = x.squeeze(1)

        # Most pretrained Wav2Vec2 checkpoints expect 16 kHz audio, while the
        # MIRACLE-AD pipeline defaults to 24 kHz. Resample in torch so this
        # optional backbone can be selected without rebuilding the data split.
        if self.sample_rate != self.model_sample_rate:
            target_length = max(
                1,
                round(x.shape[-1] * self.model_sample_rate / self.sample_rate),
            )
            x = F.interpolate(
                x.unsqueeze(1),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        # Process audio
        audio_numpy = x.detach().cpu().numpy()
        inputs = self.feature_extractor(
            audio_numpy,
            sampling_rate=self.model_sample_rate,
            padding="longest",
            return_tensors="pt"
        )

        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract features
        outputs = self.wav2vec2(
            input_values=inputs["input_values"],
            attention_mask=inputs["attention_mask"]
        )
        hidden_states = outputs.last_hidden_state

        # Mean pooling
        pooled = torch.mean(hidden_states, dim=1)  # [B, hidden_size]

        return pooled


# ==================== Classification Networks ====================

class ABMILNetwork(nn.Module):
    """ABMIL attention-based aggregation network (with optional language head)"""
    def __init__(self, feature_dim, num_classes=2, instance_hidden=256,
                 attn_hidden=128, dropout=0.3, num_language_classes=2, lang_aware=False):
        super().__init__()

        self.lang_aware = lang_aware

        # Adapter to standardize feature dimensions
        self.feature_adapter = nn.Sequential(
            nn.Linear(feature_dim, instance_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ABMIL attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(instance_hidden, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1)
        )

        # Classification head
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(instance_hidden, max(8, instance_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, instance_hidden // 2), num_classes)
        )

        # Language classification head (only used if lang_aware=True)
        if self.lang_aware:
            self.language_head = nn.Sequential(
                nn.Linear(instance_hidden, max(8, instance_hidden // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(8, instance_hidden // 2), num_language_classes)
            )

    def forward(self, x, return_attention=False, return_embeddings=False):
        """
        x: [B, N, feature_dim] - B batches, N chunks, feature_dim from backbone
        """
        # Adapt features
        inst_features = self.feature_adapter(x)  # [B, N, instance_hidden]

        # Compute attention scores
        attn_logits = self.attention(inst_features)  # [B, N, 1]
        attn_logits = attn_logits.squeeze(-1)  # [B, N]
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, N]

        # Weighted aggregation
        bag_repr = torch.sum(inst_features * attn_weights.unsqueeze(-1), dim=1)  # [B, instance_hidden]

        if return_embeddings:
            return bag_repr

        # Classification
        logits = self.classifier(bag_repr)

        if self.lang_aware:
             language_logits = self.language_head(bag_repr)
             if return_attention:
                 return (logits, language_logits), attn_weights
             return logits, language_logits

        if return_attention:
            return logits, attn_weights

        return logits

class Attn_Net_Gated(nn.Module):
    r"""
    Attention Network with Sigmoid Gating (3 fc layers)

    args:
        L (int): input feature dimension
        D (int): hidden layer dimension
        dropout (bool): whether to apply dropout (p = 0.25)
        n_classes (int): number of classes

    Formulae:
        Gated Attention Mechanism (Ilse et al., 2018):
        .. math::
            a_k = \frac{\exp\{\mathbf{w}^T (\tanh(\mathbf{V} \mathbf{h}_k^T) \odot \text{sigm}(\mathbf{U} \mathbf{h}_k^T))\}}{\sum_{j=1}^K \exp\{\mathbf{w}^T (\tanh(\mathbf{V} \mathbf{h}_j^T) \odot \text{sigm}(\mathbf{U} \mathbf{h}_j^T))\}}

    """
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        """
        x: [N, L] or [B, N, L]
        """
        # If input is 3D [B, N, L], we might need to handle it or assume batch size 1 for now
        # But typically this module expects [N, L] (bag of instances)
        # Or we can support [B, N, L] by applying linear layers carefully.

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x



class TransformerABMILNetwork(nn.Module):
    """
    Transformer ABMIL network with Gated Attention

    Formulae:
        1. Feature Extraction: $H = f_{ext}(X)$
        2. Transformation: $H' = \text{Transformer}(H)$
        3. Gated attention pooling over the transformed chunk sequence.
    """
    def __init__(self, feature_dim, num_classes=2, instance_hidden=256,
                 num_heads=8, num_layers=2, dropout=0.3, num_language_classes=2, lang_aware=False):
        super().__init__()

        self.lang_aware = lang_aware

        # Adapter to standardize feature dimensions
        self.feature_adapter = nn.Sequential(
            nn.Linear(feature_dim, instance_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.register_buffer('pos_encoding', self._get_positional_encoding(2000, instance_hidden))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=instance_hidden,
            nhead=num_heads,
            dim_feedforward=instance_hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Gated Attention Pooling
        self.attn_pool = Attn_Net_Gated(
            L=instance_hidden,
            D=instance_hidden // 2,
            dropout=True,
            n_classes=1
        )

        self.global_rho = nn.Sequential(
            nn.Linear(instance_hidden, instance_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        # Classification head
        self.classifier = nn.Linear(instance_hidden, num_classes)

        # Language classification head (only used if lang_aware=True)
        if self.lang_aware:
            self.language_head = nn.Linear(instance_hidden, num_language_classes)

    @staticmethod
    def _get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """Generate positional encoding."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x, return_attention=False, return_embeddings=False):
        """
        x: [B, N, feature_dim] - B batches, N chunks, feature_dim from backbone
        """
        B, N, _ = x.shape

        # Adapt features
        h = self.feature_adapter(x)  # [B, N, instance_hidden]

        # Add positional encoding
        seq_len = h.shape[1]
        # Ensure we don't exceed max length of PE
        if seq_len > self.pos_encoding.shape[1]:
             h = h[:, :self.pos_encoding.shape[1], :]
             seq_len = h.shape[1]

        h = h + self.pos_encoding[:, :seq_len, :]

        # Apply transformer
        h_trans = self.transformer(h)  # [B, N, instance_hidden]

        # Attention Pooling
        A, h_trans = self.attn_pool(h_trans)  # A: [B, N, 1], h_trans: [B, N, instance_hidden]
        A = torch.transpose(A, 2, 1)  # [B, 1, N]
        attn_weights = torch.nn.functional.softmax(A, dim=2)  # softmax over N [B, 1, N]

        # Weighted sum: [B, 1, N] x [B, N, L] -> [B, 1, L]
        h_path = torch.bmm(attn_weights, h_trans)
        h_path = h_path.squeeze(1) # [B, L]

        h_WSI = self.global_rho(h_path)

        if return_embeddings:
            return h_WSI

        logits = self.classifier(h_WSI)

        if self.lang_aware:
            language_logits = self.language_head(h_WSI)
            if return_attention:
                 return (logits, language_logits), attn_weights.squeeze(1) # [B, N]
            return logits, language_logits

        if return_attention:
             return logits, attn_weights.squeeze(1) # [B, N]

        return logits

class TransformerABMILNetworkWithCLS(nn.Module):
    """
    Transformer ABMIL network with CLS token and masked attention.

    Features:
    1. CLS token for knowledge accumulation
    2. Masked attention for instance-to-instance interactions with decay kernel
    3. Full attention between CLS token and all other tokens
    """
    def __init__(self, feature_dim, num_classes=2, instance_hidden=256,
                 num_heads=8, num_layers=2, dropout=0.3, num_language_classes=2,
                 lang_aware=False, decay_sigma=1.0):
        super().__init__()

        self.lang_aware = lang_aware
        self.decay_sigma = decay_sigma

        # Adapter to standardize feature dimensions
        self.feature_adapter = nn.Sequential(
            nn.Linear(feature_dim, instance_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, instance_hidden))

        # Positional encoding (for instances only, CLS gets position 0)
        self.register_buffer('pos_encoding', self._get_positional_encoding(2000, instance_hidden))

        # Custom transformer encoder with masked attention
        encoder_layer = TransformerEncoderLayerWithMask(
            d_model=instance_hidden,
            nhead=num_heads,
            dim_feedforward=instance_hidden * 4,
            dropout=dropout,
            decay_sigma=decay_sigma
        )
        self.transformer = TransformerEncoderWithMask(encoder_layer, num_layers=num_layers)

        # Classification head (uses CLS token representation)
        self.classifier = nn.Linear(instance_hidden, num_classes)

        # Optional instance-level attention for interpretability
        self.instance_attn = nn.MultiheadAttention(
            embed_dim=instance_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Language classification head (only used if lang_aware=True)
        if self.lang_aware:
            self.language_head = nn.Linear(instance_hidden, num_language_classes)

    @staticmethod
    def _get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """Generate positional encoding."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def _create_decay_mask(self, seq_len, device):
        """Create decay kernel mask for instance-to-instance attention."""
        # Create distance matrix
        positions = torch.arange(seq_len, device=device).float()
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

        # Apply Gaussian decay kernel
        decay_mask = torch.exp(-distance_matrix ** 2 / (2 * self.decay_sigma ** 2))

        return decay_mask

    def forward(self, x, return_attention=False, return_embeddings=False):
        """
        x: [B, N, feature_dim] - B batches, N chunks, feature_dim from backbone
        """
        B, N, _ = x.shape
        device = x.device

        # Adapt features
        h = self.feature_adapter(x)  # [B, N, instance_hidden]

        # Add positional encoding to instances
        seq_len = h.shape[1]
        if seq_len > self.pos_encoding.shape[1]:
            h = h[:, :self.pos_encoding.shape[1], :]
            seq_len = h.shape[1]

        h = h + self.pos_encoding[:, :seq_len, :]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, instance_hidden]
        h_with_cls = torch.cat([cls_tokens, h], dim=1)  # [B, N+1, instance_hidden]

        # Create decay mask for instance-to-instance attention
        decay_mask = self._create_decay_mask(N, device)  # [N, N]

        # Apply transformer with masked attention
        h_trans = self.transformer(h_with_cls, decay_mask)  # [B, N+1, instance_hidden]

        # Extract CLS token representation for classification
        cls_representation = h_trans[:, 0, :]  # [B, instance_hidden]

        if return_embeddings:
            return cls_representation

        # Classification using CLS token
        logits = self.classifier(cls_representation)

        # Optional: Get instance-level attention for interpretability
        if return_attention:
            # Use CLS token as query to get instance importance
            instances = h_trans[:, 1:, :]  # [B, N, instance_hidden]
            cls_query = cls_representation.unsqueeze(1)  # [B, 1, instance_hidden]

            instance_attn_weights, _ = self.instance_attn(
                cls_query, instances, instances
            )  # [B, 1, N]
            instance_attn_weights = instance_attn_weights.squeeze(1)  # [B, N]

            if self.lang_aware:
                language_logits = self.language_head(cls_representation)
                return (logits, language_logits), instance_attn_weights

            return logits, instance_attn_weights

        if self.lang_aware:
            language_logits = self.language_head(cls_representation)
            return logits, language_logits

        return logits


class TransformerEncoderLayerWithMask(nn.Module):
    """Transformer encoder layer with masked attention for decay kernel."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, decay_sigma=1.0):
        super().__init__()
        self.decay_sigma = decay_sigma
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def _scaled_dot_product_attention_with_decay(self, q, k, v, decay_mask):
        """
        Custom attention calculation with decay mask.

        Args:
            q: [B, num_heads, seq_len, head_dim]
            k: [B, num_heads, seq_len, head_dim]
            v: [B, num_heads, seq_len, head_dim]
            decay_mask: [seq_len-1, seq_len-1] - mask for instance-to-instance only

        Returns:
            attn_output: [B, seq_len, d_model]
            attn_weights: [B, num_heads, seq_len, seq_len]
        """
        B, num_heads, seq_len, head_dim = q.shape

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        # scores: [B, num_heads, seq_len, seq_len]

        # Create full attention mask
        # CLS token (position 0) can attend to all tokens with full attention
        # Instance tokens (positions 1:) can attend to CLS and each other with decay
        full_mask = torch.ones(seq_len, seq_len, device=q.device)

        if seq_len > 1:  # If there are instances
            # Apply decay mask to instance-to-instance interactions (positions 1: to 1:)
            full_mask[1:, 1:] = decay_mask

        # Expand mask to match batch and head dimensions
        full_mask = full_mask.unsqueeze(0).unsqueeze(0).expand(B, num_heads, -1, -1)

        # Apply decay mask to attention scores
        scores = scores * full_mask

        # Apply causal mask for future positions (optional, remove if not needed)
        # causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        # scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: [B, num_heads, seq_len, head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)

        return attn_output, attn_weights

    def forward(self, src, decay_mask):
        """
        src: [B, seq_len, d_model] (includes CLS token at position 0)
        decay_mask: [N, N] - mask for instance-to-instance attention only
        """
        B, seq_len, d_model = src.shape

        # Linear projections
        q = self.q_proj(src)  # [B, seq_len, d_model]
        k = self.k_proj(src)  # [B, seq_len, d_model]
        v = self.v_proj(src)  # [B, seq_len, d_model]

        # Reshape for multi-head attention
        q = q.view(B, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        # q, k, v: [B, num_heads, seq_len, head_dim]

        # Custom attention with decay mask
        attn_output, attn_weights = self._scaled_dot_product_attention_with_decay(q, k, v, decay_mask)

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection and layer norm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed forward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class TransformerEncoderWithMask(nn.Module):
    """Transformer encoder with masked attention layers."""
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, decay_mask):
        output = src
        for layer in self.layers:
            output = layer(output, decay_mask)
        return output


class GatedABMILNetwork(nn.Module):
    """
    Gated Attention ABMIL network (without Transformer) using Ilse et al. (2018) mechanism.
    """
    def __init__(self, feature_dim, num_classes=2, instance_hidden=256,
                 attn_hidden=128, dropout=0.3, num_language_classes=2, lang_aware=False):
        super().__init__()

        self.lang_aware = lang_aware

        # Adapter to standardize feature dimensions
        self.feature_adapter = nn.Sequential(
            nn.Linear(feature_dim, instance_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated Attention Mechanism
        self.attn_net = Attn_Net_Gated(
            L=instance_hidden,
            D=attn_hidden,
            dropout=True,
            n_classes=1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(instance_hidden, max(8, instance_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, instance_hidden // 2), num_classes)
        )

        # Language classification head (only used if lang_aware=True)
        if self.lang_aware:
            self.language_head = nn.Sequential(
                nn.Linear(instance_hidden, max(8, instance_hidden // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(8, instance_hidden // 2), num_language_classes)
            )

    def forward(self, x, return_attention=False, return_embeddings=False):
        """
        x: [B, N, feature_dim]
        """
        # Adapt features
        inst_features = self.feature_adapter(x)  # [B, N, instance_hidden]

        # Calculate attention scores
        attn_logits, _ = self.attn_net(inst_features)  # [B, N, 1]

        # Transpose for softmax if needed, but here eqn expects softmax over instances
        # attn_logits is [B, N, 1]. We want softmax over dim=1 (N)
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, N, 1]

        # Weighted aggregation
        # [B, N, instance_hidden] * [B, N, 1] -> [B, N, instance_hidden]
        # Sum over N -> [B, instance_hidden]
        bag_repr = torch.sum(inst_features * attn_weights, dim=1)

        if return_embeddings:
            return bag_repr

        # Classification
        logits = self.classifier(bag_repr)

        if self.lang_aware:
             language_logits = self.language_head(bag_repr)
             if return_attention:
                 return (logits, language_logits), attn_weights.squeeze(-1)
             return logits, language_logits

        if return_attention:
            return logits, attn_weights.squeeze(-1) # [B, N]

        return logits

class BiLSTMDualHeadNetwork(nn.Module):
    """BiLSTM with dual heads for disease and language classification"""
    def __init__(self, feature_dim, num_disease_classes=2, num_language_classes=2,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3, lang_aware=True):
        super().__init__()

        self.lang_aware = lang_aware

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        lstm_output_dim = lstm_hidden * 2  # bidirectional

        # Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_disease_classes)
        )

        # Language classification head (only used if lang_aware=True)
        if self.lang_aware:
            self.language_head = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_dim // 2, num_language_classes)
            )

    def forward(self, x, return_attention=False, return_embeddings=False):
        """
        x: [B, N, feature_dim] - B batches, N chunks, feature_dim from backbone
        """
        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # [B, N, lstm_hidden*2]

        # Use last time step output
        last_hidden = lstm_out[:, -1, :]  # [B, lstm_hidden*2]

        if return_embeddings:
            return last_hidden

        # Disease classification
        disease_logits = self.disease_head(last_hidden)

        if self.lang_aware:
            # Language classification
            language_logits = self.language_head(last_hidden)
            if return_attention:
               return (disease_logits, language_logits), None
            return disease_logits, language_logits

        if return_attention:
            return disease_logits, None

        return disease_logits


class UniLSTMDualHeadNetwork(nn.Module):
    """UniLSTM with dual heads for disease and language classification"""
    def __init__(self, feature_dim, num_disease_classes=2, num_language_classes=2,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3, lang_aware=True):
        super().__init__()

        self.lang_aware = lang_aware

        # UniLSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0
        )

        lstm_output_dim = lstm_hidden  # unidirectional

        # Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_disease_classes)
        )

        # Language classification head (only used if lang_aware=True)
        if self.lang_aware:
            self.language_head = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_dim // 2, num_language_classes)
            )

    def forward(self, x, return_attention=False, return_embeddings=False):
        """
        x: [B, N, feature_dim] - B batches, N chunks, feature_dim from backbone
        """
        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # [B, N, lstm_hidden]

        # Use last time step output
        last_hidden = lstm_out[:, -1, :]  # [B, lstm_hidden]

        if return_embeddings:
            return last_hidden

        # Disease classification
        disease_logits = self.disease_head(last_hidden)

        if self.lang_aware:
            # Language classification
            language_logits = self.language_head(last_hidden)
            if return_attention:
                return (disease_logits, language_logits), None
            return disease_logits, language_logits

        if return_attention:
            return disease_logits, None

        return disease_logits


# ==================== Combined Model ====================
class AudioClassificationModel(nn.Module):
    """
    Combined model with configurable backbone and classification network.

    Architecture:
    - Backbone extracts features from each audio chunk
    - Network aggregates chunk features and performs classification
    """
    def __init__(self, backbone_type: str, network_type: str,
                 num_disease_classes=2, num_language_classes=2,
                 sample_rate=24000, lang_aware=True,
                 wav2vec2_model="facebook/wav2vec2-base-960h",
                 cache_dir=None, **network_kwargs):
        super().__init__()

        if isinstance(backbone_type, BackboneType):
            backbone_type = backbone_type.value
        if isinstance(network_type, NetworkType):
            network_type = network_type.value

        self.backbone_type = backbone_type
        self.network_type = network_type
        self.lang_aware = lang_aware

        # Create backbone
        self.backbone = self._create_backbone(
            backbone_type, sample_rate, wav2vec2_model, cache_dir
        )

        # Get feature dimension from backbone
        feature_dim = self.backbone.feature_dim

        # Language classification is handled globally from the backbone
        # features, so the selected disease pooling network stays single-head.
        # NOTE: We pass lang_aware=False because we now handle language classification
        # globally in this model using average pooling of backbone features.
        self.network = self._create_network(
            network_type, feature_dim, num_disease_classes,
            num_language_classes, lang_aware=False, **network_kwargs
        )

        # Global language classification head (if lang_aware)
        if self.lang_aware:
            # Use instance_hidden if provided, else default to 256 (common in networks)
            hidden_dim = network_kwargs.get('instance_hidden', 256)
            dropout_p = network_kwargs.get('dropout', 0.3)

            self.language_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, num_language_classes)
            )

    @staticmethod
    def _create_backbone(backbone_type, sample_rate, wav2vec2_model, cache_dir):
        """Create backbone based on type."""
        backbone_map = {
            'kurdish_cnn': KurdishCNNBackbone,
            'gamma_erb_cnn': GammaERBCNNBackbone,
            'gamma_gm_cnn': GammaGMCNNBackbone,
            'soundnet8': SoundNet8Backbone,
            'soundnet5': SoundNet5Backbone,
            'm3': M3Backbone,
            'm5': M5Backbone,
            'm11': M11Backbone,
            'm18': M18Backbone,
            'raw_audio_cnn': RawAudioCNNBackbone,
            'wavenet': WaveNetBackbone,
            'raw_audio_lstm': RawAudioLSTMBackbone,
        }

        if backbone_type == 'wav2vec2':
            return Wav2Vec2Backbone(
                model_name=wav2vec2_model,
                sample_rate=sample_rate,
                cache_dir=cache_dir,
            )
        if backbone_type in backbone_map:
            return backbone_map[backbone_type](sample_rate=sample_rate)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def _create_network(self, network_type, feature_dim, num_disease_classes,
                       num_language_classes, lang_aware, **kwargs):
        """Create classification network based on type."""
        if network_type == 'abmil':
            return ABMILNetwork(
                feature_dim=feature_dim,
                num_classes=num_disease_classes,
                instance_hidden=kwargs.get('instance_hidden', 256),
                attn_hidden=kwargs.get('attn_hidden', 128),
                dropout=kwargs.get('dropout', 0.3),
                num_language_classes=num_language_classes,
                lang_aware=lang_aware
            )
        elif network_type == 'transformer_abmil':
            return TransformerABMILNetwork(
                feature_dim=feature_dim,
                num_classes=num_disease_classes,
                instance_hidden=kwargs.get('instance_hidden', 256),
                num_heads=kwargs.get('num_heads', 8),
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.3),
                num_language_classes=num_language_classes,
                lang_aware=lang_aware
            )
        elif network_type in {'bilstm', 'bilstm_dual_head'}:
            return BiLSTMDualHeadNetwork(
                feature_dim=feature_dim,
                num_disease_classes=num_disease_classes,
                num_language_classes=num_language_classes,
                lstm_hidden=kwargs.get('lstm_hidden', 128),
                lstm_layers=kwargs.get('lstm_layers', 2),
                dropout=kwargs.get('dropout', 0.3),
                lang_aware=lang_aware
            )
        elif network_type in {'unilstm', 'unilstm_dual_head'}:
            return UniLSTMDualHeadNetwork(
                feature_dim=feature_dim,
                num_disease_classes=num_disease_classes,
                num_language_classes=num_language_classes,
                lstm_hidden=kwargs.get('lstm_hidden', 128),
                lstm_layers=kwargs.get('lstm_layers', 2),
                dropout=kwargs.get('dropout', 0.3),
                lang_aware=lang_aware
            )
        elif network_type == 'gated_abmil':
            return GatedABMILNetwork(
                feature_dim=feature_dim,
                num_classes=num_disease_classes,
                instance_hidden=kwargs.get('instance_hidden', 256),
                attn_hidden=kwargs.get('attn_hidden', 128),
                dropout=kwargs.get('dropout', 0.3),
                num_language_classes=num_language_classes,
                lang_aware=lang_aware
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def forward(self, x, return_attention=False, return_embeddings=False):
        """
        Forward pass through backbone and network.

        Args:
            x: Input tensor [B, N, L] where
               B = batch size
               N = number of chunks
               L = samples per chunk
            return_attention: If True, returns (logits, attn_weights) if supported
            return_embeddings: If True, returns dictionary with embeddings {'disease': ..., 'language': ...}

        Returns:
            - If return_embeddings=True: {'disease': ..., 'language': ...}
            - If lang_aware and network supports it: (disease_logits, language_logits)
            - If return_attention=True: (logits, weights)
            - Otherwise: disease_logits
        """
        if x.dim() == 3:
            B, N, L = x.shape

            # Process each chunk through backbone
            x_flat = x.view(B * N, 1, L)  # [B*N, 1, L]
            features = self.backbone(x_flat)  # [B*N, feature_dim]

            # Reshape to [B, N, feature_dim]
            features = features.view(B, N, -1)

            # Handle Embeddings Request (Extract & Visualize Mode)
            if return_embeddings:
                # Disease Embedding (from specific network)
                # We need to ensure sub-networks support return_embeddings
                disease_emb = self.network(features, return_embeddings=True)

                # Language Embedding (General Mean Pooling)
                # This represents the "Language" vector space as defined by the language head logic
                language_emb = torch.mean(features, dim=1)

                return {
                    "disease": disease_emb,
                    "language": language_emb
                }

            network_output = self.network(
                features,
                return_attention=return_attention,
            )

            # Handle language classification
            if self.lang_aware:
                # Average pooling over chunks (dim 1)
                # features: [B, N, feature_dim] -> [B, feature_dim]
                pooled_features = torch.mean(features, dim=1)
                language_logits = self.language_head(pooled_features)

                if return_attention:
                    # network_output is expected to be (disease_logits, attn_weights)
                    # We need to unpack it carefully
                    if isinstance(network_output, tuple):
                         disease_logits = network_output[0]
                         attn_weights = network_output[1]
                    else:
                         disease_logits = network_output
                         attn_weights = None

                    return (disease_logits, language_logits), attn_weights
                else:
                    # network_output is disease_logits (since lang_aware=False for inner net)
                    # But if the inner net *always* returns tuple (it shouldn't if lang_aware=False), check it.
                    if isinstance(network_output, tuple):
                         disease_logits = network_output[0]
                    else:
                         disease_logits = network_output

                    return disease_logits, language_logits

            return network_output
        else:
            raise ValueError(f"Expected 3D input [B, N, L], got {x.dim()}D")


# ==================== Factory Function ====================

def get_model(backbone_type: str, network_type: str,
              num_disease_classes: int = 2, num_language_classes: int = 2,
              sample_rate: int = 24000, lang_aware: bool = True,
              wav2vec2_model: str = "facebook/wav2vec2-base-960h",
              cache_dir: str = None, checkpoint_path: str = None,
              **network_kwargs):
    """
    Factory function to create audio classification models.

    Args:
        backbone_type: Type of backbone ('kurdish_cnn', 'm5', 'wav2vec2', etc.)
        network_type: Type of network ('abmil', 'transformer_abmil', 'bilstm_dual_head', 'unilstm')
        num_disease_classes: Number of disease classes (default: 2)
        num_language_classes: Number of language classes (default: 2)
        sample_rate: Audio sample rate (default: 24000)
        lang_aware: Enable language awareness (default: True)
        wav2vec2_model: HuggingFace model name for Wav2Vec2 backbone
        cache_dir: Cache directory for pretrained models
        checkpoint_path: Path to checkpoint file to load
        **network_kwargs: Additional arguments for network (lstm_hidden, dropout, etc.)

    Returns:
        Configured AudioClassificationModel

    Example:
        >>> model = get_model(
        ...     backbone_type='m5',
        ...     network_type='abmil',
        ...     num_disease_classes=3,
        ...     lang_aware=False,
        ...     instance_hidden=256,
        ...     dropout=0.3
        ... )
    """
    model = AudioClassificationModel(
        backbone_type=backbone_type,
        network_type=network_type,
        num_disease_classes=num_disease_classes,
        num_language_classes=num_language_classes,
        sample_rate=sample_rate,
        lang_aware=lang_aware,
        wav2vec2_model=wav2vec2_model,
        cache_dir=cache_dir,
        **network_kwargs
    )

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)

    return model


def create_backbone(
    backbone_type: str,
    sample_rate: int = 24000,
    wav2vec2_model: str = "facebook/wav2vec2-base-960h",
    cache_dir: str = None,
):
    """Create any registered backbone without attaching a pooling head."""
    if isinstance(backbone_type, BackboneType):
        backbone_type = backbone_type.value
    return AudioClassificationModel._create_backbone(
        backbone_type, sample_rate, wav2vec2_model, cache_dir
    )


def available_backbones():
    """Return stable CLI choices for every retained backbone."""
    return [member.value for member in BackboneType]


def available_networks(include_legacy_aliases: bool = False):
    """Return stable CLI choices for every retained pooling network."""
    choices = ["unilstm", "bilstm", "abmil", "gated_abmil", "transformer_abmil"]
    if include_legacy_aliases:
        choices.extend(["unilstm_dual_head", "bilstm_dual_head"])
    return choices
