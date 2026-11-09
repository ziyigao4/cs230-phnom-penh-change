"""Run the full experiment ladder on one GPU. Resumable (skips runs already in results.jsonl).

SageMaker: data channel 'raw' -> /opt/ml/input/data/raw (zips + labels.csv)
           outputs -> /opt/ml/checkpoints (synced to S3 continuously; survives spot interruption)
"""
import argparse
import json
import os
import queue
import shutil
import sys
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (CLASSES, PatchStore, add_indices, handcrafted_features, make_split, patch_mean_series,  # noqa
                  to_float, unzip_all)

p = argparse.ArgumentParser()
p.add_argument("--raw", default=os.environ.get("SM_CHANNEL_RAW", "/opt/ml/input/data/raw"))
p.add_argument("--out", default="/opt/ml/checkpoints")
p.add_argument("--work", default="/tmp/work")
p.add_argument("--smoke", type=int, default=0)
p.add_argument("--only", default="")
p.add_argument("--seeds", default="0,1,2")
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)
os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)
LOG = open(os.path.join(args.out, "log.txt"), "a")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

from models import M1GRU, M2, PrestoClassifier, ResNet18Enc, SmallCNN  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
log("device", DEV, torch.__version__, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

# ------------------------------------------------------------------ data
npy_dir = os.path.join(args.work, "s2")
unzip_all(args.raw, npy_dir)
labels = pd.read_csv(os.path.join(args.raw, "labels.csv"))
store = PatchStore(npy_dir, limit=400 if args.smoke else None)
split = make_split(labels, store)
split.to_csv(os.path.join(args.out, "split.csv"), index=False)
log("patches", len(store.pids), "labelled", len(split), split.split.value_counts().to_dict())
log("class counts by split", split.groupby(["split", "label"]).size().to_dict())
TR = split[split.split == "train"]
VA = split[split.split == "val"]
TE = split[split.split == "test"]
ALL_ROWS = np.arange(len(store.pids))
Y_ROW = np.full(len(store.pids), -1, dtype=np.int64)
Y_ROW[split.row.values] = split.y.values

series_all = patch_mean_series(store, ALL_ROWS)  # (N,72,10)
log("series done")

# normalisation stats for 10 image channels, from training patches
_tr = add_indices(to_float(store.X[TR.row.values[:300]]))  # (n,72,10,32,32)
CH_MEAN = _tr.mean((0, 1, 3, 4)).astype(np.float32)
CH_STD = (_tr.std((0, 1, 3, 4)) + 1e-6).astype(np.float32)
del _tr
log("ch mean", CH_MEAN.round(3).tolist(), "std", CH_STD.round(3).tolist())
S_MEAN, S_STD = series_all[TR.row.values].mean((0, 1)), series_all[TR.row.values].std((0, 1)) + 1e-6

done = set()
RES = os.path.join(args.out, "results.jsonl")
if os.path.exists(RES):
    for line in open(RES):
        done.add(json.loads(line)["run"])
log("already done:", sorted(done))


# ------------------------------------------------------------------ metrics
def metrics(y, prob, df):
    pred = prob.argmax(1)
    out = {"macro_f1": f1_score(y, pred, average="macro"), "acc": accuracy_score(y, pred)}
    for i, c in enumerate(CLASSES):
        out[f"f1_{c}"] = f1_score(y == i, pred == i)
    rng = np.random.RandomState(0)
    bs = []
    for _ in range(1000):
        ii = rng.randint(0, len(y), len(y))
        bs.append(f1_score(y[ii], pred[ii], average="macro"))
    out["f1_ci_lo"], out["f1_ci_hi"] = np.percentile(bs, [2.5, 97.5])
    nb = df.nb_diff.values
    for name, m in [("interior", nb == 0), ("boundary", nb >= 0.5)]:
        if m.sum() > 5:
            out[f"f1_{name}"] = f1_score(y[m], pred[m], average="macro")
            out[f"n_{name}"] = int(m.sum())
    out["confusion"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def save_run(run, prob_all, val_m, test_m, extra):
    pd.DataFrame({"patch_id": store.pids, **{f"p_{c}": prob_all[:, i] for i, c in enumerate(CLASSES)}}) \
        .to_csv(os.path.join(args.out, "preds", f"{run}.csv"), index=False)
    rec = {"run": run, "val": val_m, "test": test_m, **extra}
    with open(RES, "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"RESULT {run}: val F1 {val_m['macro_f1']:.3f}  test F1 {test_m['macro_f1']:.3f} "
        f"[{test_m['f1_ci_lo']:.3f},{test_m['f1_ci_hi']:.3f}] interior {test_m.get('f1_interior', float('nan')):.3f} "
        f"boundary {test_m.get('f1_boundary', float('nan')):.3f}")


# ------------------------------------------------------------------ M0
def run_m0(kind, seed):
    feats = handcrafted_features(series_all)
    Xtr, Xva, Xte = feats[TR.row.values], feats[VA.row.values], feats[TE.row.values]
    if kind == "rf":
        clf = RandomForestClassifier(500, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)
    else:
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, class_weight="balanced", random_state=seed)
    clf.fit(Xtr, TR.y)
    pa = clf.predict_proba(feats)
    return pa, metrics(VA.y.values, pa[VA.row.values], VA), metrics(TE.y.values, pa[TE.row.values], TE)


# ------------------------------------------------------------------ deep input builders (GPU)
CM = torch.tensor(CH_MEAN, device=DEV).view(1, 1, 10, 1, 1)
CS = torch.tensor(CH_STD, device=DEV).view(1, 1, 10, 1, 1)


def gpu_image(xi16, delta, raw_s2=False):
    """int16 (B,T,8,H,W) -> float (B,T,C,H,W) normalised; C=10 (+8 delta)."""
    x = xi16.to(DEV, non_blocking=True).float()
    s2 = x[:, :, :4] / 1e4
    s1 = x[:, :, 4:6] / 100.0
    cnt = x[:, :, 6:8]
    b3, b4, b8 = s2[:, :, 1:2], s2[:, :, 2:3], s2[:, :, 3:4]
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    x = torch.cat([s2, s1, cnt, ndvi, ndwi], 2)
    xn = (x - CM) / CS
    if raw_s2:  # pretrained SSL4EO branch expects reflectance/1e4
        xn = torch.cat([s2, xn[:, :, 4:]], 2)
    if delta:
        sel = [0, 1, 2, 3, 4, 5, 8, 9]
        base = (x[:, :12, sel]).mean(1, keepdim=True)
        d = (x[:, :, sel] - base) / CS[:, :, sel]
        xn = torch.cat([xn, d], 2)
    return xn


def augment(x):
    if np.random.rand() < 0.5:
        x = x.flip(-1)
    if np.random.rand() < 0.5:
        x = x.flip(-2)
    k = np.random.randint(4)
    return torch.rot90(x, k, (-2, -1)) if k else x


def gpu_series(rows, delta):
    s = torch.tensor((series_all[rows] - S_MEAN) / S_STD, device=DEV)
    if delta:
        s = torch.cat([s, s - s[:, :12].mean(1, keepdim=True)], 2)
    return s


# Presto inputs: 16 sub-patch (8x8) mean series, 17 presto bands
def build_presto_inputs():
    N = len(store.pids)
    out = np.zeros((N, 16, 72, 17), np.float32)
    for i in range(N):
        x = to_float(store.X[i])  # (72,8,32,32)
        x = x.reshape(72, 8, 4, 8, 4, 8).mean((3, 5)).reshape(72, 8, 16).transpose(2, 0, 1)  # (16,72,8)
        b2, b3, b4, b8, vv, vh = [x[..., j] for j in range(6)]
        o = out[i]
        o[..., 0], o[..., 1] = (vv + 25) / 25, (vh + 25) / 25
        o[..., 2], o[..., 3], o[..., 4], o[..., 8] = b2, b3, b4, b8
        o[..., 16] = (b8 - b4) / (b8 + b4 + 1e-6)
    return out


LATLON = None


def get_latlon():
    global LATLON
    if LATLON is None:
        ll = labels.set_index("patch_id").reindex(store.pids)[["lat", "lon"]].values.astype(np.float32)
        LATLON = torch.tensor(ll, device=DEV)
    return LATLON


PRESTO_X = None


# ------------------------------------------------------------------ generic deep trainer
class Prefetch:
    def __init__(self, fn, items, n=3):
        self.q = queue.Queue(n)
        self.t = threading.Thread(target=self._run, args=(fn, items), daemon=True)
        self.t.start()

    def _run(self, fn, items):
        for it in items:
            self.q.put(fn(it))
        self.q.put(None)

    def __iter__(self):
        while True:
            x = self.q.get()
            if x is None:
                return
            yield x


def train_deep(cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    kind, k, delta = cfg["kind"], cfg.get("k", 1), cfg.get("delta", False)
    c_in = 10 + (8 if delta else 0)
    if kind == "gru":
        model = M1GRU(20 if delta else 10)
    elif kind == "cnn":
        model = M2(SmallCNN(c_in, width=cfg.get("width", 32), stem_stride=2 if k > 1 else 1))
    elif kind == "r18":
        in_map = [1, 2, 3, 7] + [-1] * (c_in - 4)
        model = M2(ResNet18Enc(c_in, in_map, cfg["pretrained"]))
    elif kind == "presto":
        global PRESTO_X
        if PRESTO_X is None:
            PRESTO_X = build_presto_inputs()
            log("presto inputs built")
        model = PrestoClassifier(cfg["pretrained"])
    model = model.to(DEV)
    bs = cfg.get("bs", 32)
    epochs = 1 if args.smoke else cfg.get("epochs", 30)
    enc_params = [p_ for n, p_ in model.named_parameters() if n.startswith("enc.") and p_.requires_grad]
    other = [p_ for n, p_ in model.named_parameters() if not n.startswith("enc.") and p_.requires_grad]
    lr = cfg.get("lr", 1e-3)
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr * cfg.get("enc_lr_mult", 1.0)},
                             {"params": other, "lr": lr}], weight_decay=0.05)
    steps = epochs * int(np.ceil(len(TR) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                total_steps=steps, pct_start=0.1)
    cnt = np.bincount(TR.y, minlength=3)
    cw = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float32, device=DEV)
    cw = cw / cw.mean()
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")

    def fetch(rows):
        if kind in ("cnn", "r18"):
            return rows, torch.from_numpy(np.stack([store.window(i, k) for i in rows])).pin_memory() \
                if DEV == "cuda" else torch.from_numpy(np.stack([store.window(i, k) for i in rows]))
        return rows, None

    def forward(rows, xb, train):
        if kind == "gru":
            return model(gpu_series(rows, delta))
        if kind == "presto":
            return model(torch.tensor(PRESTO_X[rows], device=DEV), get_latlon()[rows])
        x = gpu_image(xb, delta, raw_s2=(kind == "r18" and cfg["pretrained"]))
        if train:
            x = augment(x)
        return model(x, k) if kind == "cnn" else model(x)

    def predict(rows):
        model.eval()
        out = []
        b = bs * 2
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            for r, xb in Prefetch(fetch, [rows[i:i + b] for i in range(0, len(rows), b)]):
                out.append(torch.softmax(forward(r, xb, False).float(), 1).cpu().numpy())
        model.train()
        return np.concatenate(out)

    best, best_state, t0 = -1, None, time.time()
    tr_rows = TR.row.values
    for ep in range(epochs):
        perm = np.random.permutation(tr_rows)
        batches = [perm[i:i + bs] for i in range(0, len(perm), bs)]
        tot = 0
        for r, xb in Prefetch(fetch, batches):
            y = torch.tensor(Y_ROW[r], device=DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                logits = forward(r, xb, True)
                loss = F.cross_entropy(logits.float(), y, weight=cw, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(r)
        pv = predict(VA.row.values)
        f1 = f1_score(VA.y.values, pv.argmax(1), average="macro")
        if f1 > best:
            best, best_ep = f1, ep
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        log(f"  ep {ep:02d} loss {tot / len(tr_rows):.4f} valF1 {f1:.3f} best {best:.3f}@{best_ep} {time.time() - t0:.0f}s")
    model.load_state_dict(best_state)
    pa = predict(ALL_ROWS)
    return pa, metrics(VA.y.values, pa[VA.row.values], VA), metrics(TE.y.values, pa[TE.row.values], TE), \
        {"best_epoch": best_ep, "train_sec": time.time() - t0,
         "n_params": sum(p_.numel() for p_ in model.parameters())}, model


# ------------------------------------------------------------------ experiment table (priority order)
EXPS = [
    ("M0_rf", {"kind": "m0", "m0": "rf"}),
    ("M0_hgb", {"kind": "m0", "m0": "hgb"}),
    ("M1_gru", {"kind": "gru", "epochs": 60, "lr": 2e-3, "bs": 64}),
    ("M1_gru_delta", {"kind": "gru", "delta": True, "epochs": 60, "lr": 2e-3, "bs": 64}),
    ("M2_cnn_k1", {"kind": "cnn", "k": 1, "epochs": 40, "lr": 1e-3}),
    ("M2_cnn_k1_delta", {"kind": "cnn", "k": 1, "delta": True, "epochs": 40, "lr": 1e-3}),
    ("M2_cnn_k3_delta", {"kind": "cnn", "k": 3, "delta": True, "epochs": 30, "lr": 1e-3, "bs": 16}),
    ("R18_scratch_delta", {"kind": "r18", "pretrained": False, "delta": True, "epochs": 25, "lr": 5e-4, "bs": 16}),
    ("R18_ssl4eo_delta", {"kind": "r18", "pretrained": True, "delta": True, "epochs": 25, "lr": 5e-4,
                          "enc_lr_mult": 0.2, "bs": 16}),
    ("Presto_scratch", {"kind": "presto", "pretrained": False, "epochs": 30, "lr": 5e-4, "bs": 32}),
    ("Presto_pretrained", {"kind": "presto", "pretrained": True, "epochs": 30, "lr": 5e-4, "enc_lr_mult": 0.2,
                           "bs": 32}),
    ("M2_cnn_k5_delta", {"kind": "cnn", "k": 5, "delta": True, "epochs": 30, "lr": 1e-3, "bs": 8}),
]
if args.only:
    EXPS = [e for e in EXPS if e[0] in args.only.split(",")]
seeds = [0] if args.smoke else [int(s) for s in args.seeds.split(",")]

for seed in seeds:  # seed 0 for every model first, so partial results are already a full table
    for name, cfg in EXPS:
        run = f"{name}_s{seed}"
        if run in done:
            continue
        log(f"=== {run} {cfg}")
        try:
            if cfg["kind"] == "m0":
                pa, vm, tm = run_m0(cfg["m0"], seed)
                save_run(run, pa, vm, tm, {"cfg": cfg, "seed": seed})
            else:
                pa, vm, tm, extra, model = train_deep(cfg, seed)
                save_run(run, pa, vm, tm, {"cfg": cfg, "seed": seed, **extra})
                if seed == 0:
                    torch.save(model.state_dict(), os.path.join(args.out, f"{run}.pt"))
                del model
                torch.cuda.empty_cache()
        except Exception as e:  # keep going; record the failure
            import traceback
            log(f"FAILED {run}: {e}\n{traceback.format_exc()}")

# ------------------------------------------------------------------ summary + seed ensemble
recs = [json.loads(line) for line in open(RES)]
rows = []
for r in recs:
    rows.append({"model": r["run"].rsplit("_s", 1)[0], "seed": r["seed"], "val_f1": r["val"]["macro_f1"],
                 **{f"test_{k}": v for k, v in r["test"].items() if k != "confusion"}})
df = pd.DataFrame(rows)
for c in ["test_f1_interior", "test_f1_boundary"]:
    if c not in df:
        df[c] = np.nan
agg = df.groupby("model").agg(n=("seed", "count"), val_f1=("val_f1", "mean"), test_f1=("test_macro_f1", "mean"),
                              test_f1_std=("test_macro_f1", "std"), test_acc=("test_acc", "mean"),
                              f1_interior=("test_f1_interior", "mean"), f1_boundary=("test_f1_boundary", "mean"),
                              f1_urban=("test_f1_urban", "mean"), f1_seasonal=("test_f1_seasonal", "mean"),
                              f1_stable=("test_f1_stable", "mean"))
# seed-ensembles
ens = []
for m in agg.index:
    fs = [os.path.join(args.out, "preds", f"{m}_s{s}.csv") for s in df[df.model == m].seed]
    fs = [f for f in fs if os.path.exists(f)]
    if len(fs) > 1:
        P = np.mean([pd.read_csv(f)[[f"p_{c}" for c in CLASSES]].values for f in fs], 0)
        ens.append({"model": m + "_ENS", "test_f1": f1_score(TE.y.values, P[TE.row.values].argmax(1), average="macro"),
                    "val_f1": f1_score(VA.y.values, P[VA.row.values].argmax(1), average="macro"), "n": len(fs)})
agg = pd.concat([agg.reset_index(), pd.DataFrame(ens)], ignore_index=True)
agg.sort_values("test_f1", ascending=False).to_csv(os.path.join(args.out, "summary.csv"), index=False)
log("\n" + agg.sort_values("test_f1", ascending=False).round(3).to_string(index=False))
if os.path.isdir("/opt/ml/model"):
    for f in ["summary.csv", "results.jsonl", "split.csv", "log.txt"]:
        shutil.copy(os.path.join(args.out, f), "/opt/ml/model/")
log("ALL DONE")
