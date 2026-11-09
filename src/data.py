"""Data loading for the Phnom Penh monthly S2/S1 patch dataset.

Raw patch file: phnom_penh_{col}_{row}.npy, int16, shape (72, 8, 32, 32)
  ch0-3: S2 B2,B3,B4,B8 (reflectance*1e4, 0 = no valid obs)
  ch4-5: S1 VV,VH (dB*100)
  ch6  : S2 valid-obs count that month
  ch7  : S1 obs count that month
72 months assumed Jan 2017 .. Dec 2022.
"""
import glob
import os
import zipfile

import numpy as np
import pandas as pd

CLASSES = ["stable", "seasonal", "urban"]
C2I = {c: i for i, c in enumerate(CLASSES)}
T = 72
COL0, ROW0, NCOL, NROW = 1499, 3954, 80, 80


def unzip_all(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for z in sorted(glob.glob(os.path.join(raw_dir, "*.zip"))):
        print("unzip", z, flush=True)
        with zipfile.ZipFile(z) as f:
            for m in f.namelist():
                if m.endswith(".npy"):
                    dst = os.path.join(out_dir, os.path.basename(m))
                    if not os.path.exists(dst):
                        with f.open(m) as src, open(dst, "wb") as o:
                            o.write(src.read())
    # also allow loose npy files in raw_dir (or raw_dir/s2)
    for p in glob.glob(os.path.join(raw_dir, "**", "*.npy"), recursive=True):
        dst = os.path.join(out_dir, os.path.basename(p))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(p), dst)


def pid_to_cr(pid):
    c, r = pid.split("_")[-2:]
    return int(c), int(r)


class PatchStore:
    """All available patches in one int16 array + grid index (for neighbour windows)."""

    def __init__(self, npy_dir, limit=None):
        files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        if limit:
            files = files[:limit]
        self.pids = [os.path.basename(f)[:-4] for f in files]
        n = len(files)
        print(f"loading {n} patches", flush=True)
        self.X = np.zeros((n, T, 8, 32, 32), dtype=np.int16)
        for i, f in enumerate(files):
            a = np.load(f)
            self.X[i] = a[:, :8]
        self.grid = -np.ones((NCOL + 4, NROW + 4), dtype=np.int64)  # +2 pad each side
        for i, p in enumerate(self.pids):
            c, r = pid_to_cr(p)
            self.grid[c - COL0 + 2, r - ROW0 + 2] = i
        self.idx = {p: i for i, p in enumerate(self.pids)}
        self._fill_cloudy()

    def _fill_cloudy(self):
        """Forward- then backward-fill S2 pixels with no valid obs (all four bands == 0)."""
        X = self.X
        for t in range(1, T):
            bad = (X[:, t, :4] == 0).all(1, keepdims=True)
            X[:, t, :4] = np.where(bad, X[:, t - 1, :4], X[:, t, :4])
        for t in range(T - 2, -1, -1):
            bad = (X[:, t, :4] == 0).all(1, keepdims=True)
            X[:, t, :4] = np.where(bad, X[:, t + 1, :4], X[:, t, :4])

    def window(self, i, k):
        """Return (T,8,32k,32k) int16 window of k x k patches centred on patch i (k odd)."""
        if k == 1:
            return self.X[i]
        c, r = pid_to_cr(self.pids[i])
        gc, gr = c - COL0 + 2, r - ROW0 + 2
        h = k // 2
        out = np.zeros((T, 8, 32 * k, 32 * k), dtype=np.int16)
        for dc in range(-h, h + 1):
            for dr in range(-h, h + 1):
                j = self.grid[gc + dc, gr + dr] if 0 <= gc + dc < self.grid.shape[0] and 0 <= gr + dr < self.grid.shape[1] else -1
                if j >= 0:
                    # verified by seam continuity: grid row r+1 lies ABOVE row r; col c+1 lies to the right
                    y0, x0 = (h - dr) * 32, (dc + h) * 32
                    out[:, :, y0:y0 + 32, x0:x0 + 32] = self.X[j]
        return out


def make_split(labels_df, store, block=8, seed=0, frac=(0.70, 0.15, 0.15), buffer=1):
    """Spatial block split over the 80x80 grid; train/val patches within `buffer` of a test patch are dropped."""
    df = labels_df[labels_df.label.isin(CLASSES)].copy()
    df = df[df.patch_id.isin(store.idx)]
    cr = np.array([pid_to_cr(p) for p in df.patch_id])
    df["gc"], df["gr"] = cr[:, 0] - COL0, cr[:, 1] - ROW0
    df["block"] = (df.gc // block) * 100 + (df.gr // block)
    blocks = np.array(sorted(df.block.unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(blocks)
    n = len(blocks)
    ntr, nva = int(round(frac[0] * n)), int(round(frac[1] * n))
    bsplit = {b: "train" for b in blocks[:ntr]}
    bsplit.update({b: "val" for b in blocks[ntr:ntr + nva]})
    bsplit.update({b: "test" for b in blocks[ntr + nva:]})
    df["split"] = df.block.map(bsplit)
    # buffer
    test_mask = np.zeros((NCOL + 2 * buffer, NROW + 2 * buffer), bool)
    for gc, gr in df.loc[df.split == "test", ["gc", "gr"]].values:
        test_mask[gc:gc + 2 * buffer + 1, gr:gr + 2 * buffer + 1] = True
    near = np.array([test_mask[gc + buffer, gr + buffer] for gc, gr in df[["gc", "gr"]].values])
    df.loc[(df.split != "test") & near, "split"] = "buffer"
    df["y"] = df.label.map(C2I)
    df["row"] = df.patch_id.map(store.idx)
    # boundary flag: fraction of 8-neighbours with a different (non-drop) label
    lab = {(a, b): l for a, b, l in df[["gc", "gr", "label"]].values}
    allab = {}
    for p, l in labels_df[["patch_id", "label"]].values:
        c, r = pid_to_cr(p)
        allab[(c - COL0, r - ROW0)] = l
    diff = []
    for gc, gr, l in df[["gc", "gr", "label"]].values:
        nb = [allab.get((gc + a, gr + b)) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b)]
        nb = [x for x in nb if x in CLASSES]
        diff.append(np.mean([x != l for x in nb]) if nb else np.nan)
    df["nb_diff"] = diff
    return df


# ----------------------------------------------------------------- feature helpers
def to_float(x):
    """int16 (...,8,H,W) -> float32 physical units: S2 refl [0,1], S1 dB, counts."""
    x = x.astype(np.float32)
    out = np.empty_like(x)
    out[..., :4, :, :] = x[..., :4, :, :] / 1e4
    out[..., 4:6, :, :] = x[..., 4:6, :, :] / 100.0
    out[..., 6:8, :, :] = x[..., 6:8, :, :]
    return out


def add_indices(x):
    """x: (...,8,H,W) physical -> (...,10,H,W) adding NDVI, NDWI."""
    b3, b4, b8 = x[..., 1:2, :, :], x[..., 2:3, :, :], x[..., 3:4, :, :]
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    return np.concatenate([x, ndvi, ndwi], axis=-3)


def patch_mean_series(store, rows):
    """(N,72,10) spatial means: B2,B3,B4,B8,VV,VH,S2cnt,S1cnt,NDVI,NDWI."""
    out = np.zeros((len(rows), T, 10), np.float32)
    for k, i in enumerate(rows):
        x = add_indices(to_float(store.X[i]))
        out[k] = x.mean((-1, -2))
    return out


def handcrafted_features(series):
    """series (N,72,C) -> (N, C*15) annual/trend/seasonality stats (M0)."""
    N, T_, C = series.shape
    yr = series.reshape(N, 6, 12, C)
    ann = yr.mean(2)  # N,6,C
    t = np.arange(T_)
    tc = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    slope = np.einsum("ntc,t->nc", series, tc)
    feats = [series.mean(1), np.percentile(series, 10, axis=1), np.percentile(series, 90, axis=1),
             ann[:, 0], ann[:, -1], ann[:, -1] - ann[:, 0], slope, series.std(1), yr.std(2).mean(1)]
    feats += [ann[:, k] for k in range(6)]
    return np.concatenate(feats, 1)
