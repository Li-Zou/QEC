import os
os.chdir(r'F:\projects\QEC')

import stim
import pymatching
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 0. General utility functions
# ==========================================
def build_grid_mapper(circuit):
    """Given a circuit's coordinates, return (det_to_grid, T, X, Y)"""
    coords = circuit.get_detector_coordinates()
    coords_array = np.array([coords[i] for i in range(circuit.num_detectors)])

    # Stim ordering: (x, y, t)
    T = int(coords_array[:, 2].max()) + 1
    X = int(coords_array[:, 0].max()) + 1
    Y = int(coords_array[:, 1].max()) + 1

    det_to_grid = {}
    for i in range(circuit.num_detectors):
        x, y, t = coords[i]
        det_to_grid[i] = (int(t), int(x), int(y))
    return det_to_grid, T, X, Y

def to_grid(detection_events, det_to_grid, T, X, Y):
    """(shots, num_detectors) -> (shots, T, X, Y)"""
    shots = detection_events.shape[0]
    grid = np.zeros((shots, T, X, Y), dtype=np.float32)
    for det_idx, (t, x, y) in det_to_grid.items():
        grid[:, t, x, y] = detection_events[:, det_idx]
    return grid

# Fix a global coordinate system
circuit_base = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=3, rounds=3,
    after_clifford_depolarization=0.003, before_round_data_depolarization=0.003,
    before_measure_flip_probability=0.003, after_reset_flip_probability=0.003,
)
DET_TO_GRID, T, X, Y = build_grid_mapper(circuit_base)

# ==========================================
# 1. Data generation: data for 4 fixed p values (for training)
# ==========================================
print("=== Step 1: Generate training data ===")
p_values = [0.001, 0.003, 0.005, 0.007]

for p in p_values:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p,
    )
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots=100_000, separate_observables=True)

    grid_data = to_grid(det, DET_TO_GRID, T, X, Y)
    X_data, y_data = grid_data, obs.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    np.savez_compressed(f"surface_d3_p{p}_train.npz", X=X_train, y=y_train)
    np.savez_compressed(f"surface_d3_p{p}_test.npz", X=X_test, y=y_test)
    print(f"Saved data for p={p}, grid shape: {grid_data.shape}")

# ==========================================
# 2. Define CNN model and training function
# ==========================================
class SurfaceCodeCNN(nn.Module):
    def __init__(self, in_channels=1, t_dim=4, x_dim=7, y_dim=7):
        super().__init__()
        # Improvement 1: remove pooling and directly extract sparse features with fully connected layers
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Compute flattened dimension
        flat_size = 64 * t_dim * x_dim * y_dim

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)   # (batch, 1, T, X, Y)
        x = self.features(x)
        return self.classifier(x)

def load_data(p):
    train = np.load(f"surface_d3_p{p}_train.npz")
    test = np.load(f"surface_d3_p{p}_test.npz")
    return (torch.tensor(train["X"]), torch.tensor(train["y"]),
            torch.tensor(test["X"]), torch.tensor(test["y"]))

def train_model(p, epochs=15, batch_size=256, lr=1e-3):
    print(f"\n=== Step 2: Train CNN model (p={p}) ===")
    X_train, y_train, X_test, y_test = load_data(p)

    # Improvement 2: increase positive-class weight to prevent predicting all zeros
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).cuda() if torch.cuda.is_available() else torch.tensor([num_neg / num_pos])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurfaceCodeCNN(t_dim=T, x_dim=X, y_dim=Y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Weighted loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        # Measure test LER each epoch
        model.eval()
        with torch.no_grad():
            logits = model(X_test.to(device)).cpu()
            preds = (logits > 0).float()
            ler = (preds != y_test).any(dim=1).float().mean().item()
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(X_train):.4f} | Test LER: {ler:.4e}")

    return model

# Train the model on p=0.005
model = train_model(p=0.005, epochs=15)
_, _, X_test_005, _ = load_data(0.005)

# ==========================================
# 3. Noise drift robustness test
# ==========================================
print("\n=== Step 3: Noise drift robustness test ===")

def generate_drift_dataset(p_start, p_end, n_segments=10, shots_per_segment=10000):
    all_det, all_obs, all_p, matchings = [], [], [], []
    p_values_drift = np.linspace(p_start, p_end, n_segments)

    for p in p_values_drift:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=3, rounds=3,
            after_clifford_depolarization=p, before_round_data_depolarization=p,
            before_measure_flip_probability=p, after_reset_flip_probability=p,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        matchings.append(pymatching.Matching.from_detector_error_model(dem))

        sampler = circuit.compile_detector_sampler()
        det, obs = sampler.sample(shots=shots_per_segment, separate_observables=True)
        all_det.append(det)
        all_obs.append(obs)
        all_p.append(np.full(shots_per_segment, p))

    return np.concatenate(all_det), np.concatenate(all_obs), p_values_drift, matchings

det_drift, obs_drift, p_drift_vals, matchings = generate_drift_dataset(
    p_start=0.003, p_end=0.007, n_segments=10, shots_per_segment=10000
)
n_seg = len(p_drift_vals)
seg_size = len(det_drift) // n_seg

# 3.1 MWPM (fixed prior p=0.005)
matching_fixed = matchings[5] # center point is p=0.005
pred_fixed = matching_fixed.decode_batch(det_drift)
ler_fixed = np.any(pred_fixed != obs_drift, axis=1)
ler_per_seg_fixed = [ler_fixed[i*seg_size:(i+1)*seg_size].mean() for i in range(n_seg)]

# 3.2 MWPM (oracle perfect prior)
ler_per_seg_oracle = []
for i, matching in enumerate(matchings):
    seg_det = det_drift[i*seg_size:(i+1)*seg_size]
    seg_obs = obs_drift[i*seg_size:(i+1)*seg_size]
    pred = matching.decode_batch(seg_det)
    ler_per_seg_oracle.append(np.any(pred != seg_obs, axis=1).mean())

# 3.3 CNN inference
model.eval()
grid_drift = to_grid(det_drift, DET_TO_GRID, T, X, Y)

all_cnn_preds = []
with torch.no_grad():
    for i in range(0, len(grid_drift), 10000):
        batch_x = torch.tensor(grid_drift[i:i+10000]).to(model.parameters().__next__().device)
        logits = model(batch_x).cpu()
        all_cnn_preds.append((logits > 0).float().numpy())
preds_cnn = np.concatenate(all_cnn_preds)
ler_cnn = np.any(preds_cnn != obs_drift, axis=1)
ler_per_seg_cnn = [ler_cnn[i*seg_size:(i+1)*seg_size].mean() for i in range(n_seg)]

# ==========================================
# 4. Plotting
# ==========================================
print("\n=== Step 4: Plot comparison figure ===")
plt.figure(figsize=(8, 5))
plt.plot(p_drift_vals, ler_per_seg_fixed, 'o-', label='MWPM (fixed prior p=0.005)', color='red')
plt.plot(p_drift_vals, ler_per_seg_oracle, 's--', label='MWPM (oracle per-segment)', color='gray')
plt.plot(p_drift_vals, ler_per_seg_cnn, '^-', label='CNN (trained on p=0.005)', color='blue')
plt.xlabel('True physical error rate p (drifting)')
plt.ylabel('Logical error rate')
plt.yscale('log')
plt.legend()
plt.title('Noise Drift Robustness: d=3 Surface Code')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('noise_drift_comparison.png', dpi=150)
print("Figure saved to noise_drift_comparison.png")
plt.show()