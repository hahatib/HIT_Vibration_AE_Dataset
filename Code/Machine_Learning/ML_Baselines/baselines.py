import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from scipy.stats import entropy
from PyEMD import EMD
import pywt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emd = EMD()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_emd(signal):
    try:
        IMFs = emd(signal, max_imf=3)
    except Exception:
        IMFs = np.array([signal])
    num_imfs = 3
    emd_features = {}
    for i in range(num_imfs):
        if i < len(IMFs):
            imf_component = IMFs[i]
            energy = np.sum(np.square(imf_component)) / len(imf_component)
            emd_features[f"IMF{i + 1}_Energy"] = energy
        else:
            emd_features[f"IMF{i + 1}_Energy"] = 0
            emd_features[f"IMF{i + 1}_Kurtosis"] = 0
            emd_features[f"IMF{i + 1}_Skewness"] = 0
            emd_features[f"IMF{i + 1}_Ratio"] = 0
            emd_features[f"IMF{i + 1}_Corr"] = 0
    return emd_features


def extract_wpd(signal, wavelet="db8", level=3):
    wpd_features = {}
    wp = pywt.WaveletPacket(
        data=signal, wavelet=wavelet, mode="symmetric", maxlevel=level
    )
    nodes = [n for n in wp.get_level(level, order="freq")]
    energy_values = [np.sum(node.data**2) for node in nodes]
    total_energy = sum(energy_values) + 1e-09
    energy_ratios = [e / total_energy for e in energy_values]
    wpd_features["WPD_Energy_Entropy"] = entropy(energy_ratios)
    for i, node in enumerate(nodes):
        data = node.data
        node_id = f"WPD_L{level}_Node{i}"
        wpd_features[f"{node_id}_Mean"] = np.mean(data)
        rms_val = np.sqrt(np.mean(data**2))
        max_val = np.max(np.abs(data))
        if rms_val == 0:
            crest_val = 0.0
        else:
            crest_val = max_val / rms_val
        wpd_features[f"{node_id}_CrestFactor"] = crest_val
    return wpd_features


def extract_features(signal, mode):
    signal = np.asarray(signal)
    features_dict = {"峰峰值": np.max(signal) - np.min(signal)}
    if mode == "EMD":
        features_dict.update(extract_emd(signal))
    elif mode == "WPD":
        features_dict.update(extract_wpd(signal))
    return features_dict


def get_feature_matrices(X_slice):
    X_slice = np.asarray(X_slice)
    pairs = [
        (extract_features(signal, mode="EMD"), extract_features(signal, mode="WPD"))
        for signal in X_slice
    ]
    emd_rows = [pair[0] for pair in pairs]
    wpd_rows = [pair[1] for pair in pairs]
    return (pd.DataFrame(emd_rows).values, pd.DataFrame(wpd_rows).values)


class SimpleANN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(SimpleANN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        features = self.relu(x)
        out = self.classifier(features)
        return (out, features)


class DeepDNN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(DeepDNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return (out, features)


def train_evaluate_ml(
    X_train,
    y_train,
    X_test,
    model_type,
    num_classes,
    epochs=50,
    seed=42,
):
    global model
    set_seed(seed)
    X_train_s = np.asarray(X_train)
    X_test_s = np.asarray(X_test)
    preds = []
    probabilities = None
    features_for_tsne = None
    if model_type == "SVM":
        model = SVC(kernel="rbf", C=0.1, probability=True, random_state=int(seed))
        print("[stage] svm_fit", flush=True)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        probabilities = model.predict_proba(X_test_s)
        features_for_tsne = X_test_s
    elif model_type in ["ANN", "DNN"]:
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
        train_ds = TensorDataset(X_train_t, y_train_t)
        generator = torch.Generator().manual_seed(int(seed))
        train_loader = DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        input_dim = X_train.shape[1]
        if model_type == "ANN":
            model = SimpleANN(input_dim, num_classes).to(device)
        elif model_type == "DNN":
            model = DeepDNN(input_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(epochs):
            print(f"[epoch] epoch={epoch + 1}/{epochs}", flush=True)
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                out, _ = model(inputs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            outputs, feats = model(X_test_t.to(device))
            _, preds_t = torch.max(outputs, 1)
            preds = preds_t.cpu().numpy()
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            features_for_tsne = feats.cpu().numpy()
    return {
        "preds": preds,
        "probabilities": probabilities,
        "features": features_for_tsne,
    }
