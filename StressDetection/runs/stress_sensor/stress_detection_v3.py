"""
StressDetectionV3 — Cow Stress Early Warning System
=====================================================
Predicts stress 2 hours ahead using:
  - THI (Temperature-Humidity Index)
  - Neck body temperature
  - Lying behavior

Output: 3 classes → 0=Normal, 1=At-Risk, 2=Stressed
"""


"""Time-series input
      ↓
BiLSTM (reads past patterns)
      ↓
Attention (weighs which sensor matters) a neural network component that allows models to dynamically focus on relevant parts of input data
      ↓
Classifier (outputs Normal / At-Risk / Stressed)
"""
"PUT THE THRESHOLDS HERE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# THRESHOLDS  (fill in your own values)
# ─────────────────────────────────────────────
THI_AT_RISK_THRESHOLD    = 72   # e.g. 72
THI_STRESSED_THRESHOLD   = 78   # e.g. 78

NECK_AT_RISK_THRESHOLD   = 38.5   # e.g. 38.5  (°C)x    
NECK_STRESSED_THRESHOLD  = 39.5   # e.g. 39.5  (°C)

LYING_MIN_MINUTES        = 8   # e.g. 8   (hours/day → convert to your unit)
LYING_MAX_MINUTES        = 14  # e.g. 14  (hours/day → convert to your unit)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LABEL GENERATOR
#     Call this on your raw data to build the ground-truth labels.
#     Stress is defined by ANY of the 3 rules being triggered.
# ─────────────────────────────────────────────────────────────────────────────
def generate_stress_label(thi: float,
                           neck_temp: float,
                           lying_minutes: float,
                           neck_temp_slope: float) -> int:
    """
    Returns:
        0 = Normal
        1 = At-Risk   (early warning)
        2 = Stressed  (confirmed stress)

    neck_temp_slope: rate of change of neck temp per minute
                     (positive = rising, exponential rise → high value)
    """
    score = 0  # accumulate severity

    # Rule 1 — THI
    if THI_STRESSED_THRESHOLD and thi >= THI_STRESSED_THRESHOLD:
        score += 2
    elif THI_AT_RISK_THRESHOLD and thi >= THI_AT_RISK_THRESHOLD:
        score += 1

    # Rule 2 — Lying behavior (too much OR too little)
    if LYING_MIN_MINUTES and LYING_MAX_MINUTES:
        if lying_minutes < LYING_MIN_MINUTES or lying_minutes > LYING_MAX_MINUTES:
            score += 1

    # Rule 3 — Body temperature + exponential rise pattern
    if NECK_STRESSED_THRESHOLD and neck_temp >= NECK_STRESSED_THRESHOLD:
        score += 2
    elif NECK_AT_RISK_THRESHOLD and neck_temp >= NECK_AT_RISK_THRESHOLD:
        score += 1

    # Exponential rise bonus (slope-based)
    if neck_temp_slope is not None and neck_temp_slope > 0.05:   # rising fast → bump up
        score += 1

    # Map score to class
    if score >= 4:
        return 2   # Stressed
    elif score >= 2:
        return 1   # At-Risk
    else:
        return 0   # Normal
# ─────────────────────────────────────────────────────────────────────────────
# 2.  SENSOR ENCODERS
#     Each sensor gets its own BiLSTM encoder + projection.
#     Input shape: (batch, seq_len, input_size)
# ─────────────────────────────────────────────────────────────────────────────
class SensorEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for a single sensor stream.
    Also computes the rate-of-change (slope) as an extra feature,
    which is crucial for detecting exponential temperature rise.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size + 1,      # +1 for slope/delta feature
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        # Compute delta (rate of change) and append as feature
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]   # first difference
        x_aug = torch.cat([x, delta], dim=-1)            # (B, T, input_size+1)

        out, _ = self.lstm(x_aug)                        # (B, T, hidden*2)

        # Use both last forward and last backward hidden states
        last = out[:, -1, :]                             # (B, hidden*2)
        return self.proj(last)                           # (B, output_size)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CROSS-SENSOR ATTENTION FUSION
#     Learns which sensor to trust most at each moment.
#     e.g. at night neck temp matters more; during hot days THI matters more.
# ─────────────────────────────────────────────────────────────────────────────
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, num_sensors: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, sensor_embeddings: list) -> torch.Tensor:
        """
        sensor_embeddings: list of (B, D) tensors, one per sensor
        Returns: (B, D) fused representation
        """
        # Stack → (B, num_sensors, D)
        x = torch.stack(sensor_embeddings, dim=1)
        # Self-attention across sensors
        attended, _ = self.attn(x, x, x)
        attended = self.norm(attended + x)               # residual
        # Mean pool across sensors
        return attended.mean(dim=1)                      # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
class StressDetectionV3(nn.Module):
    """
    Multi-modal early stress detection for dairy cows.

    Inputs (all as time-series windows of length `seq_len`):
        thi         : (B, seq_len, 1)  — THI values
        neck_temp   : (B, seq_len, 1)  — Neck/body temperature
        lying       : (B, seq_len, 1)  — Lying duration per interval
        cow_id      : (B,)             — Integer cow identity

    Output:
        logits      : (B, 3)           — [Normal, At-Risk, Stressed]

    The model predicts stress 2 HOURS AHEAD:
        → your DataLoader must shift labels by 2h relative to the input window.
    """

    # Sensor output embedding dimension (all projected to same size for fusion)
    EMBED_DIM = 64

    def __init__(self,
                 num_cows: int = 20,
                 cow_embed_dim: int = 16,
                 dropout: float = 0.3):
        super().__init__()

        D = self.EMBED_DIM

        # ── Sensor Encoders ──────────────────────────────────────────────────
        self.thi_enc = SensorEncoder(
            input_size=1, hidden_size=32, num_layers=2,
            output_size=D, dropout=dropout
        )
        self.neck_enc = SensorEncoder(
            input_size=1, hidden_size=64, num_layers=2,
            output_size=D, dropout=dropout
        )
        self.lying_enc = SensorEncoder(
            input_size=1, hidden_size=32, num_layers=2,
            output_size=D, dropout=dropout
        )

        # ── Cow Identity Embedding ───────────────────────────────────────────
        self.cow_emb = nn.Embedding(num_cows, cow_embed_dim)

        # ── Attention Fusion (3 sensors) ─────────────────────────────────────
        self.fusion = AttentionFusion(embed_dim=D, num_sensors=3)

        # ── Final Classifier ─────────────────────────────────────────────────
        fused_dim = D + cow_embed_dim          # 64 + 16 = 80
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 3)                   # 3 classes
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                thi: torch.Tensor,
                neck_temp: torch.Tensor,
                lying: torch.Tensor,
                cow_id: torch.Tensor) -> torch.Tensor:

        # Encode each sensor independently
        e_thi   = self.thi_enc(thi)           # (B, 64)
        e_neck  = self.neck_enc(neck_temp)    # (B, 64)
        e_lying = self.lying_enc(lying)       # (B, 64)

        # Attention fusion across sensors
        fused = self.fusion([e_thi, e_neck, e_lying])   # (B, 64)

        # Append cow identity
        cow_vec = self.cow_emb(cow_id)                  # (B, 16)
        combined = torch.cat([fused, cow_vec], dim=-1)  # (B, 80)

        return self.classifier(combined)                # (B, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATASET
#     Sliding window dataset — labels are shifted 2 hours into the future.
# ─────────────────────────────────────────────────────────────────────────────
class CowStressDataset(torch.utils.data.Dataset):
    """
    Args:
        thi_arr    : np.ndarray (T,)   — THI time series
        neck_arr   : np.ndarray (T,)   — Neck temperature time series
        lying_arr  : np.ndarray (T,)   — Lying duration time series
        labels_arr : np.ndarray (T,)   — Stress labels (0/1/2) for each timestep
        cow_id     : int               — Cow identifier
        window     : int               — Input window length (timesteps)
        horizon    : int               — How many steps ahead to predict
                                         e.g. if 1 step = 1 min → horizon = 120
    """
    def __init__(self, thi_arr, neck_arr, lying_arr, labels_arr,
                 cow_id: int, window: int = 60, horizon: int = 120):
        self.thi    = torch.FloatTensor(thi_arr).unsqueeze(-1)
        self.neck   = torch.FloatTensor(neck_arr).unsqueeze(-1)
        self.lying  = torch.FloatTensor(lying_arr).unsqueeze(-1)
        self.labels = torch.LongTensor(labels_arr)
        self.cow_id = cow_id
        self.window  = window
        self.horizon = horizon
        self.length  = len(thi_arr) - window - horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s, e = idx, idx + self.window
        label_idx = e + self.horizon - 1       # label 2 hours ahead

        return (
            self.thi[s:e],                     # (window, 1)
            self.neck[s:e],                    # (window, 1)
            self.lying[s:e],                   # (window, 1)
            torch.tensor(self.cow_id),         # scalar
            self.labels[label_idx]             # scalar label (0/1/2)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train(model, dataloader, optimizer, device, class_weights=None):
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_loss, correct, total = 0, 0, 0

    for thi, neck, lying, cow_id, labels in dataloader:
        thi, neck, lying = thi.to(device), neck.to(device), lying.to(device)
        cow_id, labels   = cow_id.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(thi, neck, lying, cow_id)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — important for LSTMs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device, class_weights=None):
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for thi, neck, lying, cow_id, labels in dataloader:
        thi, neck, lying = thi.to(device), neck.to(device), lying.to(device)
        cow_id, labels   = cow_id.to(device), labels.to(device)

        logits = model(thi, neck, lying, cow_id)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# 7.  QUICK START — wire everything together
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Instantiate model ────────────────────────────────────────────────────
    model = StressDetectionV3(
        num_cows=12,          # update to your total number of cows
        cow_embed_dim=16,
        dropout=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Class weights (handle imbalance — stressed samples are rare) ─────────
    # Adjust these based on your actual class distribution
    # e.g. if 70% normal, 20% at-risk, 10% stressed:
    class_weights = torch.tensor([1.0, 3.0, 6.0]).to(device)

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # ── Dummy forward pass to verify shapes ─────────────────────────────────
    B, T = 4, 60   # batch=4, window=60 timesteps
    dummy_thi    = torch.randn(B, T, 1).to(device)
    dummy_neck   = torch.randn(B, T, 1).to(device)
    dummy_lying  = torch.randn(B, T, 1).to(device)
    dummy_cow    = torch.randint(0, 12, (B,)).to(device)

    out = model(dummy_thi, dummy_neck, dummy_lying, dummy_cow)
    print(f"Output shape: {out.shape}")   # Expected: (4, 3)
    print(f"Predicted classes: {out.argmax(dim=-1)}")

    # ── Save model ───────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "StressDetectionV3.pt")
    print("Model saved as StressDetectionV3.pt")
