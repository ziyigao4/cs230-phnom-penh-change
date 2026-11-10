"""Inference / evaluation of the trained models on NEW patches (e.g. a human-labelled gold set).

Two sub-commands:

  export   Run once on the ORIGINAL training data. Recomputes the exact normalisation statistics used in
           training (same split + same code path as run_all.py), refits the M0 baselines and saves them, then
           self-checks: re-predicts the original test split with the saved .pt weights and compares against the
           probabilities written during training.
             python src/predict.py export --raw data/raw --weights results/checkpoints --preds results/preds \
                    --out artifacts

  predict  Score new patches with every model found in the artifacts dir.
             python src/predict.py predict --data new_patches/ --labels gold.csv --artifacts artifacts \
                    --out eval_gold
           new_patches/: <prefix>_{col}_{row}.npy files, int16 (72, 8, 32, 32), same export script / channels /
                         scaling as training, 72 months starting in January.
           gold.csv:     columns patch_id, label (stable|seasonal|urban; anything else, e.g. "uncertain", is kept
                         in the prediction file but not scored); optional lat, lon (needed by Presto; otherwise
                         the Phnom Penh centre is used and a warning is printed).
           Outputs: <out>/preds_<model>.csv, <out>/metrics.json, <out>/METRICS.md
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (CLASSES, C2I, PatchStore, add_indices, handcrafted_features, make_split,  # noqa: E402
                  patch_mean_series, to_float, unzip_all)

# must mirror EXPS in run_all.py (only the fields needed to rebuild the network / inputs)
MODEL_CFGS = {
    "M1_gru": {"kind": "gru", "delta": False},
    "M1_gru_delta": {"kind": "gru", "delta": True},
    "M2_cnn_k1": {"kind": "cnn", "k": 1, "delta": False},
    "M2_cnn_k1_delta": {"kind": "cnn", "k": 1, "delta": True},
    "M2_cnn_k3_delta": {"kind": "cnn", "k": 3, "delta": True},
    "M2_cnn_k5_delta": {"kind": "cnn", "k": 5, "delta": True},
    "R18_scratch_delta": {"kind": "r18", "pretrained": False, "delta": True},
    "R18_ssl4eo_delta": {"kind": "r18", "pretrained": True, "delta": True},
    "Presto_scratch": {"kind": "presto", "pretrained": False},
    "Presto_pretrained": {"kind": "presto", "pretrained": True},
}
PP_LATLON = (11.55, 104.92)


# ------------------------------------------------------------------ inputs (identical maths to run_all.py)
def presto_inputs(store):
    N = len(store.pids)
    out = np.zeros((N, 16, 72, 17), np.float32)
    for i in range(N):
        x = to_float(store.X[i])
        x = x.reshape(72, 8, 4, 8, 4, 8).mean((3, 5)).reshape(72, 8, 16).transpose(2, 0, 1)
        b2, b3, b4, b8, vv, vh = [x[..., j] for j in range(6)]
        o = out[i]
        o[..., 0], o[..., 1] = (vv + 25) / 25, (vh + 25) / 25
        o[..., 2], o[..., 3], o[..., 4], o[..., 8] = b2, b3, b4, b8
        o[..., 16] = (b8 - b4) / (b8 + b4 + 1e-6)
    return out


class Predictor:
    def __init__(self, stats, device):
        import torch
        self.torch, self.dev = torch, device
        self.CM = torch.tensor(stats["ch_mean"], device=device).view(1, 1, 10, 1, 1)
        self.CS = torch.tensor(stats["ch_std"], device=device).view(1, 1, 10, 1, 1)
        self.S_MEAN = np.array(stats["s_mean"], np.float32)
        self.S_STD = np.array(stats["s_std"], np.float32)

    def gpu_image(self, xi16, delta, raw_s2=False):
        torch = self.torch
        x = xi16.to(self.dev).float()
        s2 = x[:, :, :4] / 1e4
        s1 = x[:, :, 4:6] / 100.0
        cnt = x[:, :, 6:8]
        b3, b4, b8 = s2[:, :, 1:2], s2[:, :, 2:3], s2[:, :, 3:4]
        ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
        ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
        x = torch.cat([s2, s1, cnt, ndvi, ndwi], 2)
        xn = (x - self.CM) / self.CS
        if raw_s2:
            xn = torch.cat([s2, xn[:, :, 4:]], 2)
        if delta:
            sel = [0, 1, 2, 3, 4, 5, 8, 9]
            base = (x[:, :12, sel]).mean(1, keepdim=True)
            xn = torch.cat([xn, (x[:, :, sel] - base) / self.CS[:, :, sel]], 2)
        return xn

    def build(self, name):
        from models import M1GRU, M2, PrestoClassifier, ResNet18Enc, SmallCNN
        cfg = MODEL_CFGS[name]
        delta, k = cfg.get("delta", False), cfg.get("k", 1)
        c_in = 10 + (8 if delta else 0)
        if cfg["kind"] == "gru":
            return M1GRU(20 if delta else 10)
        if cfg["kind"] == "cnn":
            return M2(SmallCNN(c_in, width=32, stem_stride=2 if k > 1 else 1))
        if cfg["kind"] == "r18":  # weights come from the .pt; no need to download SSL4EO again
            return M2(ResNet18Enc(c_in, [1, 2, 3, 7] + [-1] * (c_in - 4), pretrained=False))
        return PrestoClassifier(pretrained=False)

    def predict(self, name, weight_path, store, rows, latlon, bs=32):
        torch = self.torch
        cfg = MODEL_CFGS[name]
        model = self.build(name)
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        model.to(self.dev).eval()
        kind, k, delta = cfg["kind"], cfg.get("k", 1), cfg.get("delta", False)
        if kind == "gru":
            series = patch_mean_series(store, rows)
        if kind == "presto":
            px = presto_inputs(store)
        out = []
        amp = self.dev == "cuda"
        with torch.no_grad(), torch.autocast(self.dev, dtype=torch.float16, enabled=amp):
            for s in range(0, len(rows), bs):
                r = rows[s:s + bs]
                if kind == "gru":
                    x = (series[s:s + bs] - self.S_MEAN) / self.S_STD
                    x = torch.tensor(x, device=self.dev)
                    if delta:
                        x = torch.cat([x, x - x[:, :12].mean(1, keepdim=True)], 2)
                    logits = model(x)
                elif kind == "presto":
                    logits = model(torch.tensor(px[r], device=self.dev), torch.tensor(latlon[r], device=self.dev))
                else:
                    xb = torch.from_numpy(np.stack([store.window(i, k) for i in r]))
                    x = self.gpu_image(xb, delta, raw_s2=(kind == "r18" and cfg["pretrained"]))
                    logits = model(x, k) if kind == "cnn" else model(x)
                out.append(torch.softmax(logits.float(), 1).cpu().numpy())
        return np.concatenate(out)


# ------------------------------------------------------------------ metrics
def metrics(y, prob):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    pred = prob.argmax(1)
    out = {"n": int(len(y)), "macro_f1": float(f1_score(y, pred, average="macro")),
           "acc": float(accuracy_score(y, pred))}
    for i, c in enumerate(CLASSES):
        out[f"f1_{c}"] = float(f1_score(y == i, pred == i, zero_division=0))
    rng = np.random.RandomState(0)
    bs = [f1_score(y[ii], pred[ii], average="macro") for ii in (rng.randint(0, len(y), len(y)) for _ in range(1000))]
    out["f1_ci_lo"], out["f1_ci_hi"] = [float(v) for v in np.percentile(bs, [2.5, 97.5])]
    out["confusion"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    return out


# ------------------------------------------------------------------ export
def cmd_export(a):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    os.makedirs(a.out, exist_ok=True)
    npy_dir = os.path.join(a.work, "s2")
    unzip_all(a.raw, npy_dir)
    labels = pd.read_csv(os.path.join(a.raw, "labels.csv"))
    store = PatchStore(npy_dir)
    split = make_split(labels, store)
    TR, TE = split[split.split == "train"], split[split.split == "test"]
    _tr = add_indices(to_float(store.X[TR.row.values[:300]]))
    series_all = patch_mean_series(store, np.arange(len(store.pids)))
    stats = {"ch_mean": _tr.mean((0, 1, 3, 4)).astype(np.float32).tolist(),
             "ch_std": (_tr.std((0, 1, 3, 4)) + 1e-6).astype(np.float32).tolist(),
             "s_mean": series_all[TR.row.values].mean((0, 1)).tolist(),
             "s_std": (series_all[TR.row.values].std((0, 1)) + 1e-6).tolist(),
             "classes": CLASSES, "months": "72, starting January (assumed 2017-01..2022-12)",
             "note": "ch_*: B2,B3,B4,B8,VV,VH,S2cnt,S1cnt,NDVI,NDWI computed as in run_all.py"}
    del _tr
    json.dump(stats, open(os.path.join(a.out, "norm_stats.json"), "w"), indent=1)
    print("saved norm_stats.json", flush=True)
    feats = handcrafted_features(series_all)
    for name, clf in [("M0_rf", RandomForestClassifier(500, min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
                                                       random_state=0)),
                      ("M0_hgb", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                                class_weight="balanced", random_state=0))]:
        clf.fit(feats[TR.row.values], TR.y)
        joblib.dump(clf, os.path.join(a.out, f"{name}.joblib"), compress=3)
        print("saved", name, flush=True)
    # copy deep weights next to the stats
    import shutil
    for pt in glob.glob(os.path.join(a.weights, "*_s0.pt")):
        shutil.copy(pt, a.out)
    # self-check against training-time predictions on the original test split
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = Predictor(stats, dev)
    rows = TE.row.values
    ll = labels.set_index("patch_id").reindex(store.pids)[["lat", "lon"]].values.astype(np.float32)
    check = {}
    for pt in sorted(glob.glob(os.path.join(a.out, "*_s0.pt"))):
        name = os.path.basename(pt)[:-6]
        if name not in MODEL_CFGS:
            continue
        prob = P.predict(name, pt, store, rows, ll)
        ref_f = os.path.join(a.preds, f"{name}_s0.csv")
        ref = pd.read_csv(ref_f).set_index("patch_id").loc[TE.patch_id, [f"p_{c}" for c in CLASSES]].values
        check[name] = {"max_abs_diff_vs_training": float(np.abs(prob - ref).max()),
                       "test_f1_reproduced": metrics(TE.y.values, prob)["macro_f1"],
                       "test_f1_training": metrics(TE.y.values, ref)["macro_f1"]}
        print(name, check[name], flush=True)
    for name in ["M0_rf", "M0_hgb"]:
        clf = joblib.load(os.path.join(a.out, f"{name}.joblib"))
        check[name] = {"test_f1_reproduced": metrics(TE.y.values, clf.predict_proba(feats[rows]))["macro_f1"]}
    json.dump(check, open(os.path.join(a.out, "export_selfcheck.json"), "w"), indent=1)
    print("EXPORT DONE", flush=True)


# ------------------------------------------------------------------ predict
def cmd_predict(a):
    import joblib
    import torch
    os.makedirs(a.out, exist_ok=True)
    npy_dir = a.data
    if glob.glob(os.path.join(a.data, "*.zip")):
        npy_dir = os.path.join(a.work, "new")
        unzip_all(a.data, npy_dir)
    store = PatchStore(npy_dir)
    stats = json.load(open(os.path.join(a.artifacts, "norm_stats.json")))
    lab = pd.read_csv(a.labels) if a.labels else pd.DataFrame({"patch_id": store.pids})
    lab = lab.drop_duplicates("patch_id").set_index("patch_id")
    missing = [p for p in lab.index if p not in store.idx]
    if missing:
        print(f"WARNING {len(missing)} labelled patches have no .npy file, e.g. {missing[:5]}", flush=True)
    lab = lab[lab.index.isin(store.idx)]
    rows = np.array([store.idx[p] for p in lab.index])
    if {"lat", "lon"} <= set(lab.columns):
        ll_lab = lab[["lat", "lon"]].values.astype(np.float32)
    else:
        print("WARNING no lat/lon columns; Presto uses the Phnom Penh centre", flush=True)
        ll_lab = np.tile(np.array(PP_LATLON, np.float32), (len(lab), 1))
    latlon = np.tile(np.array(PP_LATLON, np.float32), (len(store.pids), 1))
    latlon[rows] = ll_lab
    has_y = "label" in lab.columns
    scored = lab.label.isin(CLASSES).values if has_y else np.zeros(len(lab), bool)
    y = lab.label[scored].map(C2I).values if has_y else None
    if has_y:
        print(f"{len(lab)} patches, {scored.sum()} scorable; label counts {lab.label.value_counts().to_dict()}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = Predictor(stats, dev)
    names = a.models.split(",") if a.models else None
    results, probs = {}, {}
    # M0
    feats = None
    for name in ["M0_rf", "M0_hgb"]:
        f = os.path.join(a.artifacts, f"{name}.joblib")
        if os.path.exists(f) and (names is None or name in names):
            if feats is None:
                feats = handcrafted_features(patch_mean_series(store, rows))
            probs[name] = joblib.load(f).predict_proba(feats)
    # deep
    for name in MODEL_CFGS:
        f = os.path.join(a.artifacts, f"{name}_s0.pt")
        if os.path.exists(f) and (names is None or name in names):
            print("predicting", name, flush=True)
            probs[name] = P.predict(name, f, store, rows, latlon)
    for name, pr in probs.items():
        df = pd.DataFrame({"patch_id": lab.index, **{f"p_{c}": pr[:, i] for i, c in enumerate(CLASSES)},
                           "pred": [CLASSES[i] for i in pr.argmax(1)]})
        if has_y:
            df["label"] = lab.label.values
        df.to_csv(os.path.join(a.out, f"preds_{name}.csv"), index=False)
        if has_y and scored.sum():
            results[name] = metrics(y, pr[scored])
    if has_y and scored.sum():
        json.dump(results, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
        md = [f"# Evaluation on `{os.path.basename(os.path.normpath(a.data))}` ({scored.sum()} scored patches)", "",
              "| Model | Macro-F1 | 95% CI | Acc | F1 stable | F1 seasonal | F1 urban |", "|---|---|---|---|---|---|---|"]
        for n, m in sorted(results.items(), key=lambda kv: -kv[1]["macro_f1"]):
            md.append(f"| `{n}` | **{m['macro_f1']:.3f}** | [{m['f1_ci_lo']:.3f}, {m['f1_ci_hi']:.3f}] | {m['acc']:.3f} | "
                      f"{m['f1_stable']:.3f} | {m['f1_seasonal']:.3f} | {m['f1_urban']:.3f} |")
        md += ["", "Confusion matrices (rows = true, cols = pred; stable, seasonal, urban):", ""]
        md += [f"- `{n}`: {m['confusion']}" for n, m in results.items()]
        open(os.path.join(a.out, "METRICS.md"), "w").write("\n".join(md) + "\n")
        print("\n".join(md), flush=True)
    print("PREDICT DONE", flush=True)


if __name__ == "__main__":
    # SageMaker passes hyper-parameters only as "--key value": allow "--mode export|predict" instead of a positional
    if len(sys.argv) > 1 and sys.argv[1] not in ("export", "predict", "-h", "--help"):
        mode = "export"
        if "--mode" in sys.argv:
            i = sys.argv.index("--mode")
            mode = sys.argv[i + 1]
            del sys.argv[i:i + 2]
        sys.argv.insert(1, mode)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--raw", default=os.environ.get("SM_CHANNEL_RAW", "/opt/ml/input/data/raw"))
    e.add_argument("--weights", default=os.environ.get("SM_CHANNEL_CKPT", "/opt/ml/input/data/ckpt"))
    e.add_argument("--preds", default=None)
    e.add_argument("--out", default="/opt/ml/model")
    e.add_argument("--work", default="/tmp/work")
    q = sub.add_parser("predict")
    q.add_argument("--data", default=os.environ.get("SM_CHANNEL_NEW"))
    q.add_argument("--labels", default=None, help="csv path, or a file name inside --data")
    q.add_argument("--artifacts", default=os.environ.get("SM_CHANNEL_ARTIFACTS"))
    q.add_argument("--models", default="")
    q.add_argument("--out", default="eval")
    q.add_argument("--work", default="/tmp/work")
    a, _ = p.parse_known_args()
    if a.cmd == "export":
        a.preds = a.preds or os.path.join(a.weights, "preds")
        cmd_export(a)
    else:
        if a.labels and not os.path.exists(a.labels) and a.data and os.path.exists(os.path.join(a.data, a.labels)):
            a.labels = os.path.join(a.data, a.labels)
        cmd_predict(a)
