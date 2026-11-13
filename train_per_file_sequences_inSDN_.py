#!/usr/bin/env python3
"""
=====================================================================
BASE evaluation (before Cross-Scenario): per-file chronological split (InSDN)
Every file is split 80% / 20% in time order; windows never span two files.
All seeds (0..14), five models (multi-task BiLSTM-Autoencoder, BiLSTM, LSTM, XGBoost, Random Forest),
confusion matrices saved.
=====================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout,
    RepeatVector, TimeDistributed, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:                       # pip install xgboost
    XGBClassifier = None
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ====================== CONFIG ======================
DATASET_PATH = os.environ.get("INSDN_PATH", r"D:\proj\InSDN_DatasetCSV")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results_initial_evaluation")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "initial_evaluation_all_seeds.csv")

SEQUENCE_LENGTHS = [3, 5, 8, 10]

N_SEEDS = 15
SEEDS = list(range(N_SEEDS))              # every seed is run: 0 ... 14
# proposed multi-task BiLSTM-AE and the baselines (BiLSTM without reconstruction, LSTM, XGBoost, Random Forest)
MODELS = ['multitask', 'bilstm', 'lstm', 'xgboost', 'random_forest']
STRIDE = 2
EPOCHS = 40
PATIENCE = 7
RECONSTRUCTION_WEIGHT = 0.5
LEARNING_RATE = 1e-4
VAL_RATIO = 0.15                          # chronological tail of every training file

if os.environ.get("QUICK_TEST"):          # only to check that the script runs (tiny and fast)
    N_SEEDS, EPOCHS, PATIENCE = 2, 2, 1
    SEEDS = list(range(N_SEEDS))
    SEQUENCE_LENGTHS = [3]

BATCH_SIZE = 128
TEST_RATIO = 0.2

sns.set_theme(style="whitegrid")


def load_and_clean_file(filepath, label):
    print(f"  Loading: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    df['label'] = label

    if 'Timestamp' in df.columns:
        # FIX: the InSDN files contain two day-first formats ("5/2/2020 13:58" and "31/12/2019 02:25:57 PM").
        # A plain pd.to_datetime() turned the second format into NaT and those rows were silently dropped.
        n_before = len(df)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True, format='mixed')
        df = df.dropna(subset=['Timestamp'])
        df = df.sort_values('Timestamp', kind='mergesort').reset_index(drop=True)
        print(f"    Sorted by Timestamp | Samples: {len(df)} (dropped for bad timestamp: {n_before - len(df)})")

    # FIX: ports are removed as well (as described in the paper)
    cols_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Label']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    print(f"    Samples after cleaning: {len(df)} | features: {df.shape[1] - 1}")
    return df

# ====================== COMMON FUNCTIONS ======================
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def align_columns(dfs):
    """Keep only the feature columns that exist in every file (same order in all files)."""
    common = set.intersection(*[set(d.columns) for d in dfs])
    cols = [c for c in dfs[0].columns if c in common]
    return [d[cols].copy() for d in dfs]


def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(inputs)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3))(x)
    latent = Dropout(0.3)(x)

    cls = Dense(64, activation='relu')(latent)
    cls = Dropout(0.3)(cls)
    cls_out = Dense(1, activation='sigmoid', name='classification')(cls)

    repeated = RepeatVector(input_shape[0])(latent)
    dec = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(repeated)
    dec = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(dec)
    recon_out = TimeDistributed(Dense(input_shape[1], activation='linear'), name='reconstruction')(dec)

    model = Model(inputs=inputs, outputs=[cls_out, recon_out])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss={'classification': 'binary_crossentropy', 'reconstruction': 'mse'},
        loss_weights={'classification': 1.0, 'reconstruction': RECONSTRUCTION_WEIGHT},
        metrics={'classification': ['accuracy']}
    )
    return model


def build_bilstm_model(input_shape):
    """BiLSTM baseline: same encoder and classification head as the hybrid model,
    but WITHOUT the reconstruction branch and without the reconstruction loss."""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(inputs)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3))(x)
    latent = Dropout(0.3)(x)

    cls = Dense(64, activation='relu')(latent)
    cls = Dropout(0.3)(cls)
    cls_out = Dense(1, activation='sigmoid', name='classification')(cls)

    model = Model(inputs=inputs, outputs=cls_out)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_lstm_model(input_shape):
    """LSTM baseline: unidirectional LSTM encoder of the same size, classification head only."""
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True, dropout=0.3)(inputs)
    x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=False, dropout=0.3)(x)
    latent = Dropout(0.3)(x)

    cls = Dense(64, activation='relu')(latent)
    cls = Dropout(0.3)(cls)
    cls_out = Dense(1, activation='sigmoid', name='classification')(cls)

    model = Model(inputs=inputs, outputs=cls_out)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_tree_model(model_name, seed):
    """Random Forest / XGBoost: they receive every window flattened into one vector (T x F values)."""
    if model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method='hist',
                         n_jobs=-1, random_state=seed, eval_metric='logloss')


def create_sequences(df, window_size, stride=STRIDE):
    feature_cols = [c for c in df.columns if c != 'label']
    features = df[feature_cols].values.astype(np.float32)
    labels = df['label'].values

    X_seq, y_seq = [], []
    for i in range(0, len(features) - window_size + 1, stride):
        window_labels = labels[i:i + window_size]
        attack_ratio = np.mean(window_labels)
        window_label = 1 if attack_ratio > 0.5 else 0
        X_seq.append(features[i:i + window_size])
        y_seq.append(window_label)

    if len(X_seq) == 0:
        return np.empty((0, window_size, features.shape[1])), np.empty((0,))
    return np.array(X_seq), np.array(y_seq)


def split_train_val(df, val_ratio=VAL_RATIO):
    # FIX: the validation set is the chronological tail of EVERY training file.
    # (validation_split=0.15 of Keras takes the last 15% of the concatenated array, which contained
    #  only attack sequences, so early stopping looked at one class only.)
    split_idx = int(len(df) * (1 - val_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def make_sequences(dfs, seq_len):
    X_list, y_list = [], []
    for df in dfs:
        X, y = create_sequences(df, seq_len)
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def build_datasets(train_dfs, test_dfs, seq_len):
    """Windows are built per file (never across files); scaling is fitted on the training windows only."""
    tr_dfs, va_dfs = [], []
    for df in train_dfs:
        a, b = split_train_val(df)
        tr_dfs.append(a)
        va_dfs.append(b)

    X_train, y_train = make_sequences(tr_dfs, seq_len)
    X_val, y_val = make_sequences(va_dfs, seq_len)
    X_test, y_test = make_sequences(test_dfs, seq_len)

    if len(np.unique(y_val)) < 2:
        print("    WARNING: the validation set contains a single class!")
    print(f"    Train seqs: {X_train.shape[0]} | Val seqs: {X_val.shape[0]} "
          f"(attack {y_val.mean()*100:.1f}%) | Test seqs: {X_test.shape[0]} (attack {y_test.mean()*100:.1f}%)")

    n_features = X_train.shape[2]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    X_train, X_val, X_test = [np.clip(a, -10, 10) for a in (X_train, X_val, X_test)]
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(model, model_name, X_train, y_train, X_val, y_val):
    if model_name == 'multitask':
        targets = {'classification': y_train, 'reconstruction': X_train}
        val_targets = {'classification': y_val, 'reconstruction': X_val}
        monitor = 'val_classification_accuracy'
    else:
        targets, val_targets = y_train, y_val
        monitor = 'val_accuracy'

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=PATIENCE,
        restore_best_weights=True,
        mode='max',
        verbose=0
    )
    history = model.fit(
        X_train, targets,
        validation_data=(X_val, val_targets),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )
    best_epoch = int(np.argmax(history.history[monitor])) + 1
    return history, best_epoch


def plot_confusion_matrix(cm, title, path, decimals=0):
    """Confusion matrix with counts and row percentages (TN/FP/FN/TP)."""
    plt.figure(figsize=(6.5, 5.5))
    cm = np.asarray(cm, dtype=float)
    cm_perc = cm / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    annot = np.empty(cm.shape, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]:.{decimals}f}\n({cm_perc[i, j]:.1f}%)"
    ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True, linewidths=1.2, linecolor='gray',
                     xticklabels=['Normal (0)', 'Attack (1)'], yticklabels=['Normal (0)', 'Attack (1)'],
                     annot_kws={"size": 12})
    for (i, j), lab in {(0, 0): 'TN', (0, 1): 'FP', (1, 0): 'FN', (1, 1): 'TP'}.items():
        ax.text(j + 0.5, i + 0.18, lab, ha='center', va='center', color='red', fontsize=11, fontweight='bold')
    plt.title(title, fontsize=12, pad=12)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def predict_probs(model, model_name, X_test):
    """Returns (attack probability, reconstruction or None, total inference time in seconds)."""
    if model_name in ('xgboost', 'random_forest'):
        X_flat = X_test.reshape(len(X_test), -1)
        model.predict_proba(X_flat[:BATCH_SIZE])                              # warm-up (not timed)
        start_inf = time.time()
        y_prob = model.predict_proba(X_flat)[:, 1]
        return y_prob, None, time.time() - start_inf
    model.predict(X_test[:BATCH_SIZE], verbose=0, batch_size=BATCH_SIZE)      # warm-up (not timed)
    start_inf = time.time()
    out = model.predict(X_test, verbose=0, batch_size=BATCH_SIZE)
    inference_time = time.time() - start_inf
    if model_name == 'multitask':
        return out[0].flatten(), out[1], inference_time
    return out.flatten(), None, inference_time


def evaluate_model(model, model_name, X_test, y_test, seq_len, exp_name, seed):
    y_prob, recon, inference_time_total = predict_probs(model, model_name, X_test)
    inference_time_per_sample = (inference_time_total / len(X_test)) * 1000  # ms
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if recon is not None:
        recon_error = np.mean(np.square(X_test - recon), axis=(1, 2))
        normal_error = np.mean(recon_error[y_test == 0]) if np.any(y_test == 0) else 0.0
        attack_error = np.mean(recon_error[y_test == 1]) if np.any(y_test == 1) else 0.0
    else:
        normal_error = attack_error = np.nan

    # per-run plots only for the first seed (the mean confusion matrix over all seeds is plotted at the end)
    if seed == SEEDS[0]:
        tag = f"{exp_name}_{model_name}_seq{seq_len}"
        plot_confusion_matrix(cm, f'Confusion Matrix - {exp_name} - {model_name} (seq={seq_len}, seed={seed})',
                              os.path.join(RESULTS_DIR, "plots", f"cm_{tag}_seed{seed}.png"))
        if len(np.unique(y_test)) > 1:
            fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr_curve, tpr_curve, label=f'AUC = {auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.title(f'ROC - {exp_name} - {model_name} (seq={seq_len})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "plots", f"roc_{tag}_seed{seed}.png"), dpi=300)
            plt.close()

    return {
        'experiment': exp_name,
        'model': model_name,
        'sequence_length': seq_len,
        'seed': seed,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc': auc,
        'fpr': fpr,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'recon_error_normal': normal_error,
        'recon_error_attack': attack_error,
        'inference_time_total_sec': inference_time_total,
        'inference_time_per_sample_ms': inference_time_per_sample,
        'n_test': len(y_test)
    }


def run_one_experiment(data, exp_name, seq_len, seed, model_name):
    X_train, y_train, X_val, y_val, X_test, y_test = data
    n_features = X_train.shape[2]
    print(f"\n  → {exp_name} | {model_name} | seq={seq_len} | seed={seed}")

    set_seed(seed)
    if model_name in ('xgboost', 'random_forest'):
        model = build_tree_model(model_name, seed)
        start_train = time.time()
        model.fit(X_train.reshape(len(X_train), -1), y_train)
        training_time = time.time() - start_train
        epochs_trained, best_epoch = np.nan, np.nan
    else:
        if model_name == 'multitask':
            model = build_hybrid_model((seq_len, n_features))
        elif model_name == 'bilstm':
            model = build_bilstm_model((seq_len, n_features))
        else:
            model = build_lstm_model((seq_len, n_features))
        start_train = time.time()
        history, best_epoch = train_model(model, model_name, X_train, y_train, X_val, y_val)
        training_time = time.time() - start_train
        epochs_trained = len(history.history['loss'])

    metrics = evaluate_model(model, model_name, X_test, y_test, seq_len, exp_name, seed)
    metrics['training_time_sec'] = training_time
    metrics['epochs_trained'] = epochs_trained
    metrics['best_epoch'] = best_epoch

    print(f"    Acc: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1_score']*100:.2f}% | "
          f"Recall: {metrics['recall']*100:.2f}% | FPR: {metrics['fpr']*100:.2f}% | "
          f"TN/FP/FN/TP: {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']} | "
          f"Train: {training_time:.1f}s | best epoch: {best_epoch}")

    del model
    tf.keras.backend.clear_session()
    return metrics


def load_done():
    if not os.path.isfile(RESULTS_CSV):
        return set()
    d = pd.read_csv(RESULTS_CSV)
    return set(zip(d['experiment'], d['model'], d['sequence_length'], d['seed']))


def save_result(row):
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode='a', header=not os.path.isfile(RESULTS_CSV), index=False)


def run_all(experiments):
    """experiments = [(name, train_dfs, test_dfs), ...]. Every finished run is saved at once, so an
    interrupted job can simply be started again (finished runs are skipped)."""
    done = load_done()
    models = [m for m in MODELS if not (m == 'xgboost' and XGBClassifier is None)]
    if len(models) < len(MODELS):
        print("WARNING: xgboost is not installed (pip install xgboost) -> the XGBoost baseline is skipped.")
    for exp_name, train_dfs, test_dfs in experiments:
        print("\n" + "#" * 95)
        print(f"EXPERIMENT: {exp_name}  |  seeds: {SEEDS[0]}..{SEEDS[-1]}  |  models: {models}")
        print("#" * 95)
        for seq_len in SEQUENCE_LENGTHS:
            data = None
            for seed in SEEDS:
                for model_name in models:
                    if (exp_name, model_name, seq_len, seed) in done:
                        continue
                    if data is None:
                        print(f"\n  Building datasets | {exp_name} | seq={seq_len}")
                        data = build_datasets(train_dfs, test_dfs, seq_len)
                    save_result(run_one_experiment(data, exp_name, seq_len, seed, model_name))


def summarize():
    df = pd.read_csv(RESULTS_CSV)
    group = ['experiment', 'model', 'sequence_length']
    cols = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'fpr', 'tn', 'fp', 'fn', 'tp',
            'training_time_sec', 'inference_time_per_sample_ms', 'best_epoch']
    summary = df.groupby(group)[cols].agg(['mean', 'std'])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.insert(3, 'n_seeds', df.groupby(group)['seed'].nunique().values)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_mean_std.csv"), index=False)

    # Confusion matrices: mean ± std over the seeds + mean confusion-matrix plots
    cm_rows = []
    for (exp, model, L), g in df.groupby(group):
        m = {c: g[c].mean() for c in ('tn', 'fp', 'fn', 'tp')}
        s = {c: (g[c].std() if len(g) > 1 else 0.0) for c in ('tn', 'fp', 'fn', 'tp')}
        cm_rows.append({'experiment': exp, 'model': model, 'sequence_length': L, 'n_seeds': len(g),
                        'TN_mean': m['tn'], 'TN_std': s['tn'], 'FP_mean': m['fp'], 'FP_std': s['fp'],
                        'FN_mean': m['fn'], 'FN_std': s['fn'], 'TP_mean': m['tp'], 'TP_std': s['tp']})
        plot_confusion_matrix([[m['tn'], m['fp']], [m['fn'], m['tp']]],
                              f'Mean Confusion Matrix - {exp} - {model} (seq={L}, {len(g)} seeds)',
                              os.path.join(RESULTS_DIR, "plots", f"cm_mean_{exp}_{model}_seq{L}.png"))
    pd.DataFrame(cm_rows).to_csv(os.path.join(RESULTS_DIR, "confusion_matrix_summary.csv"), index=False)

    print("\n" + "=" * 95)
    print(f"FINAL RESULTS (mean ± std over seeds)  -> {RESULTS_DIR}/summary_mean_std.csv")
    print("=" * 95)
    for _, r in summary.iterrows():
        print(f"{r['experiment']:<10} {r['model']:<14} seq={int(r['sequence_length']):<3} seeds={int(r['n_seeds']):<3} "
              f"Acc {r['accuracy_mean']*100:6.2f}±{r['accuracy_std']*100:5.2f}  "
              f"F1 {r['f1_score_mean']*100:6.2f}±{r['f1_score_std']*100:5.2f}  "
              f"Rec {r['recall_mean']*100:6.2f}±{r['recall_std']*100:5.2f}  "
              f"FPR {r['fpr_mean']*100:5.2f}±{r['fpr_std']*100:5.2f}  "
              f"AUC {r['auc_mean']:.4f}  best_ep {'-' if pd.isna(r['best_epoch_mean']) else format(r['best_epoch_mean'], '.1f')}")
    print("\n--- Confusion matrices (mean ± std of TN / FP / FN / TP) ---")
    print(pd.DataFrame(cm_rows).round(1).to_string(index=False))
    print(f"\nAll results and plots saved in: {RESULTS_DIR}/")


# ====================== MAIN ======================
def main():
    print("=" * 95)
    print("INITIAL EVALUATION - Per-File Chronological Split + No Window Leakage (InSDN)")
    print(f"windows {SEQUENCE_LENGTHS} | seeds {SEEDS[0]}..{SEEDS[-1]} | models {MODELS}")
    print("=" * 95)

    files = [
        (os.path.join(DATASET_PATH, "Normal_data.csv"), 0),
        (os.path.join(DATASET_PATH, "metasploitable-2.csv"), 1),
        (os.path.join(DATASET_PATH, "OVS.csv"), 1),
    ]

    dfs = []
    print("\n[1/2] Loading files...")
    for filepath, label in files:
        dfs.append(load_and_clean_file(filepath, label))
    dfs = align_columns(dfs)

    print("\n[2/2] Chronological split per file...")
    train_dfs, test_dfs = [], []
    for (filepath, _), df in zip(files, dfs):
        split_idx = int(len(df) * TEST_RATIO)
        train_df = df.iloc[:-split_idx].copy()
        test_df = df.iloc[-split_idx:].copy()
        print(f"  {os.path.basename(filepath)} → Train: {len(train_df)} | Test: {len(test_df)}")
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    run_all([("Initial", train_dfs, test_dfs)])
    summarize()


if __name__ == "__main__":
    main()
