"""
add_features_shrimp.py
Métriques de comportement de la CREVETTE, frame par frame, sur la fenêtre :
    [Frame_start - PRE_SECONDS  ->  frappe (Catch_frame_nb)]
  - phase "pre"  : les PRE_SECONDS secondes AVANT le début de l'activité de la seiche
                   (moins si la vidéo commence plus tard : durée réelle dans pre_duration_s)
  - phase "hunt" : Frame_start -> frappe, le même segment que l'analyse de la seiche

Repère des frames : tracking COMPLET (vidéos 01_VIDEO_Cropped, frame 0 = Start) ;
Frame_start et Catch_frame_nb (Data_cuttlfish_strike.xlsx) sont dans ce repère.
Même nettoyage que add_features_full.py (sauts DLC, inversions tête <-> milieu).

Métriques (unités dans le nom ; versions *_log = pour le clustering) :
  shrimp_speed_mm_s        vitesse du centre du corps (MidShrimp)
  shrimp_accel_mm_s2       accélération (variation du vecteur vitesse)
  shrimp_accel_peak        1 sur les pics d'accélération (départs brusques / tail-flips)
  shrimp_burst_rate        nombre de pics par seconde (fenêtre BURST_WINDOW_S)
  shrimp_forward_speed     vitesse le long de l'axe du corps : > 0 avance, < 0 recule
  shrimp_swim_dir_deg      sens de nage : 0 = vers l'avant, 180 = en arrière (NaN si immobile)
  shrimp_turn_speed_deg_s  vitesse de rotation du corps sur lui-même
  shrimp_view_cuttle       où est la seiche vue par la crevette : 0 = devant sa tête, 180 = derrière
  distance_mm              distance seiche - crevette
  cuttle_speed_mm_s        vitesse de la seiche (centre du corps), pour la comparer à celle de la crevette
"""
from pathlib import Path
import pandas as pd
import numpy as np

# ── Dossiers du dépôt (chemins RELATIFS : ce script est dans Code/) ──
REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "Data"

# ══════════════════════════════════════════════════════════════
# PARAMÈTRES — change uniquement ici
# ══════════════════════════════════════════════════════════════
TRACKING_PATH = DATA_DIR / "combined_cuttlefish_tracking.csv"        # NON inclus (662 Mo)
_LOCAL_TRACKING = Path("E:/01_Travail/CREC/cuttle_strike/combined_df/combined_cuttlefish_tracking.csv")
if not TRACKING_PATH.exists() and _LOCAL_TRACKING.exists():         # copie locale (PC de l'auteur)
    TRACKING_PATH = _LOCAL_TRACKING
EXCEL_PATH    = DATA_DIR / "Data_cuttlfish_strike.xlsx"
SAVE_TEMPLATE = str(DATA_DIR / "shrimp_features_smooth_{w}.csv")

ALL_SMOOTH_WINDOW = (1,)       # ex. (1, 10) — chaque fenêtre = un CSV
PRE_SECONDS       = 30         # durée gardée AVANT Frame_start (s)
P_MIN             = 0.6        # vraisemblance DLC minimale d'un point de la crevette (sinon interpolé)
VAR_WINDOW        = 20         # fenêtre (frames) des rotations ; 20 f = 1/3 s à 60 fps
SPEED_WINDOW      = 10         # lissage des vitesses (frames)
MAX_JUMP_PX       = 50         # déplacement max physiquement possible en 1 frame (1/60 s)
MAX_GLITCH_FRAMES = 30         # au-delà, une position "trop loin" est acceptée (vrai déplacement)
SPEED_NOISE_MM_S  = 0.5        # en dessous : bruit de tracking -> vitesse mise à 0
SWIM_MIN_SPEED    = 5.0        # sens de nage défini seulement au-dessus (mm/s)
ANGLE_NOISE_DEG   = 0.15       # rotation < 0.15°/frame = bruit -> ignorée
SHRIMP_MIN_LEN    = 0.6        # axe crevette rejeté si tête-milieu < 60 % de sa longueur habituelle
ACCEL_PEAK_MM_S2  = None       # seuil des pics d'accélération (mm/s²) ; None = percentile ACCEL_PEAK_PCT
ACCEL_PEAK_PCT    = 0.99
PEAK_MIN_GAP_S    = 0.25       # deux pics plus proches = un seul départ
BURST_WINDOW_S    = 2.0        # fenêtre du taux de pics (s)
TEST_N_ANIMALS    = None       # ex. 5 pour tester rapidement

SHRIMP_POINTS = ("HeadShrimp", "MidShrimp", "TailShrimp")
CUTTLE_POINTS = ("Apex", "TopMantle")


# ══════════════════════════════════════════════════════════════
# FONCTIONS (mêmes que add_features_full.py)
# ══════════════════════════════════════════════════════════════
def valid_track(x, y, max_jump, max_glitch):
    """Positions aberrantes : comparaison à la DERNIÈRE POSITION VALIDE, distance
    autorisée = max_jump × frames écoulées ; après max_glitch frames, acceptée."""
    ok = np.isfinite(x) & np.isfinite(y)
    first = np.flatnonzero(ok)
    if len(first) == 0:
        return ok
    lx, ly, gap = x[first[0]], y[first[0]], 1
    for i in range(first[0] + 1, len(x)):
        if not ok[i]:
            gap += 1
            continue
        if np.hypot(x[i] - lx, y[i] - ly) > max_jump * gap and gap <= max_glitch:
            ok[i] = False
            gap += 1
        else:
            lx, ly, gap = x[i], y[i], 1
    return ok


def clean_positions(df, point):
    """Positions x, y d'un point, sauts aberrants retirés puis interpolés (+ nb rejetés)."""
    xs, ys, n_bad = [], [], 0
    for _, g in df.groupby("animal_id", sort=False):
        x = g[f"{point}_x"].astype(float).copy()
        y = g[f"{point}_y"].astype(float).copy()
        bad = ~valid_track(x.to_numpy(), y.to_numpy(), MAX_JUMP_PX, MAX_GLITCH_FRAMES)
        n_bad += int(bad.sum())
        x[bad] = np.nan
        y[bad] = np.nan
        xs.append(x.interpolate("linear").ffill().bfill())
        ys.append(y.interpolate("linear").ffill().bfill())
    return pd.concat(xs).reindex(df.index), pd.concat(ys).reindex(df.index), n_bad


def smooth_group(series, window):
    """Moyenne glissante centrée, bords laissés intacts."""
    half   = window // 2
    result = series.rolling(window, center=True, min_periods=1).mean()
    if half > 0 and len(series) > 2 * half:
        result.iloc[:half]  = series.iloc[:half]
        result.iloc[-half:] = series.iloc[-half:]
    elif len(series) <= 2 * half:
        result = series.copy()
    return result


def velocity(df, x_col, y_col):
    """Vitesse (px/frame) en x et y, lissée sur SPEED_WINDOW frames, par individu."""
    g = df["animal_id"]
    def _v(col):
        d = df[col].groupby(g).diff()
        return d.groupby(g).transform(
            lambda s: s.rolling(SPEED_WINDOW, center=True, min_periods=1).mean()).fillna(0)
    return _v(x_col), _v(y_col)


def angle_between_vectors(dx1, dy1, dx2, dy2):
    """Angle SANS signe entre deux vecteurs (arccos, 0..180°)."""
    dot  = dx1 * dx2 + dy1 * dy2
    norm = np.sqrt(dx1 ** 2 + dy1 ** 2) * np.sqrt(dx2 ** 2 + dy2 ** 2)
    return np.degrees(np.arccos(np.clip(dot / (norm + 1e-9), -1.0, 1.0)))


def signed_log(s):
    """log1p qui garde le signe (+ / -)."""
    return np.sign(s) * np.log1p(s.abs())


# ── 1. Chargement + découpe [Frame_start - PRE_SECONDS -> frappe] ─
cols = ["frame", "animal_id"] + [f"{p}_{c}" for p in CUTTLE_POINTS + SHRIMP_POINTS for c in "xyl"]
raw  = pd.read_csv(TRACKING_PATH, usecols=lambda c: c in cols, on_bad_lines="skip")
raw["animal_id"] = raw["animal_id"].astype(int)
print(f"[OK] Chargé : {TRACKING_PATH.name} | {len(raw)} frames | {raw['animal_id'].nunique()} individus")

Data = pd.read_excel(EXCEL_PATH)
info = Data.set_index(Data["num"].astype(int))[
    ["Catch_frame_nb", "Frame_start", "scale_1px_mm", "Fps"]].dropna()
missing = sorted(set(raw["animal_id"]) - set(info.index))
if missing:
    print(f"[WARN] {len(missing)} individus absents de l'Excel (pas de Frame_start / frappe) -> ignorés : {missing}")
raw = raw[raw["animal_id"].isin(info.index)]
if TEST_N_ANIMALS:
    raw = raw[raw["animal_id"].isin(sorted(raw["animal_id"].unique())[:TEST_N_ANIMALS])]

info["win_start"] = np.maximum(0, info["Frame_start"] - np.round(PRE_SECONDS * info["Fps"])).astype(int)
aid = raw["animal_id"]
raw = raw[(raw["frame"] >= aid.map(info["win_start"])) & (raw["frame"] <= aid.map(info["Catch_frame_nb"]))]
raw = raw.sort_values(["animal_id", "frame"]).reset_index(drop=True)

aid = raw["animal_id"]
FPS = aid.map(info["Fps"])
MM  = aid.map(info["scale_1px_mm"])                                     # mm par pixel
raw["strike_frame"]        = aid.map(info["Catch_frame_nb"])
raw["activity_frame"]      = aid.map(info["Frame_start"])
raw["frame_rel_strike"]    = raw["frame"] - raw["strike_frame"]         # 0 = frappe, < 0 avant
raw["time_to_strike_s"]    = raw["frame_rel_strike"] / FPS
raw["time_from_activity_s"] = (raw["frame"] - raw["activity_frame"]) / FPS   # 0 = début activité seiche
raw["phase"]               = np.where(raw["frame"] < raw["activity_frame"], "pre", "hunt")
raw["pre_duration_s"]      = aid.map((info["Frame_start"] - info["win_start"]) / info["Fps"])

pre = raw.groupby("animal_id")["pre_duration_s"].first()
print(f"[OK] Fenêtre [-{PRE_SECONDS} s avant activité -> frappe] : {len(raw)} frames | {aid.nunique()} individus")
print(f"     phase pre : {(raw['phase'] == 'pre').sum()} frames | hunt : {(raw['phase'] == 'hunt').sum()} frames")
print(f"     pre < {PRE_SECONDS} s (vidéo trop courte) : {(pre < PRE_SECONDS).sum()} individus "
      f"| sans aucune frame pre : {(pre == 0).sum()}")

# ── 1b. Points de la crevette peu fiables (vraisemblance DLC < P_MIN) -> NaN ─
for p in SHRIMP_POINTS:
    low = raw[f"{p}_l"] < P_MIN
    raw.loc[low, [f"{p}_x", f"{p}_y"]] = np.nan
    print(f"[DLC] {p:<11} vraisemblance < {P_MIN} : {low.mean():.1%} des frames "
          f"(pre {low[raw['phase'] == 'pre'].mean():.1%} | hunt {low[raw['phase'] == 'hunt'].mean():.1%})")


for SMOOTH_WINDOW in ALL_SMOOTH_WINDOW:
    print(f"\n=== SMOOTH_WINDOW={SMOOTH_WINDOW} frames")
    SAVE_PATH = SAVE_TEMPLATE.format(w=SMOOTH_WINDOW)
    df = raw.copy()

    # ── 2. Nettoyage des sauts (interpole aussi les points peu fiables) puis lissage ─
    rejected = {}
    for pt in SHRIMP_POINTS + CUTTLE_POINTS:
        df[f"{pt}_x"], df[f"{pt}_y"], rejected[pt] = clean_positions(df, pt)
    print("[sauts] positions rejetées : " +
          " | ".join(f"{pt} {n} ({n / len(df):.2%})" for pt, n in rejected.items()))
    w = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW + 1
    if w > 1:
        for col in [c for c in df.columns if c.endswith(("_x", "_y"))]:
            df[col] = df.groupby("animal_id")[col].transform(lambda s: smooth_group(s, w))

    gid = df["animal_id"]

    # axe du corps de la crevette (milieu -> tête) ; inversions tête <-> milieu
    # (crevette repliée) : tête-milieu s'effondre -> axe rejeté puis interpolé
    sh_x = df["HeadShrimp_x"] - df["MidShrimp_x"]
    sh_y = df["HeadShrimp_y"] - df["MidShrimp_y"]
    sh_len  = np.sqrt(sh_x ** 2 + sh_y ** 2)
    flipped = sh_len < SHRIMP_MIN_LEN * sh_len.groupby(gid).transform("median")
    sh_x = sh_x.mask(flipped).groupby(gid).transform(lambda s: s.interpolate().ffill().bfill())
    sh_y = sh_y.mask(flipped).groupby(gid).transform(lambda s: s.interpolate().ffill().bfill())
    sh_n = np.sqrt(sh_x ** 2 + sh_y ** 2) + 1e-9
    print(f"[crevette] axe rejeté (tête-milieu trop court) : {flipped.sum()} frames ({flipped.mean():.2%})")

    # ── 3. Vitesse et accélération (centre du corps = MidShrimp) ─
    vx, vy = velocity(df, "MidShrimp_x", "MidShrimp_y")                 # px/frame
    v = np.sqrt(vx ** 2 + vy ** 2) * MM * FPS
    v = v.clip(upper=v.quantile(0.995))
    v[v < SPEED_NOISE_MM_S] = 0.0
    df["shrimp_speed_mm_s"] = v
    df["shrimp_speed_log"]  = np.log1p(v)

    #   vitesse de la SEICHE (centre du corps Apex-TopMantle), mêmes règles : comparaison
    #   seiche / crevette dans 5_shrimp_impact (qui change de vitesse en premier ?)
    df["cuttle_cx"] = (df["Apex_x"] + df["TopMantle_x"]) / 2
    df["cuttle_cy"] = (df["Apex_y"] + df["TopMantle_y"]) / 2
    cvx, cvy = velocity(df, "cuttle_cx", "cuttle_cy")
    cv = np.sqrt(cvx ** 2 + cvy ** 2) * MM * FPS
    cv = cv.clip(upper=cv.quantile(0.995))
    cv[cv < SPEED_NOISE_MM_S] = 0.0
    df["cuttle_speed_mm_s"] = cv

    #   accélération = norme de la variation du vecteur vitesse (mm/s²)
    ax_ = vx.groupby(gid).diff().fillna(0) * MM * FPS * FPS
    ay_ = vy.groupby(gid).diff().fillna(0) * MM * FPS * FPS
    acc = np.sqrt(ax_ ** 2 + ay_ ** 2)
    acc = acc.groupby(gid).transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
    df["shrimp_accel_mm_s2"] = acc.clip(upper=acc.quantile(0.995))
    df["shrimp_accel_log"]   = np.log1p(df["shrimp_accel_mm_s2"])

    #   pics : maximum local au-dessus du seuil, au moins PEAK_MIN_GAP_S entre deux pics
    thr = ACCEL_PEAK_MM_S2 if ACCEL_PEAK_MM_S2 is not None else df["shrimp_accel_mm_s2"].quantile(ACCEL_PEAK_PCT)
    peak = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("animal_id").indices.items():
        a = df["shrimp_accel_mm_s2"].to_numpy()[idx]
        gap = int(PEAK_MIN_GAP_S * FPS.iloc[idx[0]])
        cand = np.flatnonzero((a > thr) & (a >= np.roll(a, 1)) & (a >= np.roll(a, -1)))
        for c in cand[np.argsort(-a[cand])]:                             # les plus forts d'abord
            if np.all(np.abs(np.flatnonzero(peak[idx]) - c) >= gap):
                peak[idx[c]] = 1
    df["shrimp_accel_peak"] = peak
    bw = int(round(BURST_WINDOW_S * FPS.median()))
    df["shrimp_burst_rate"] = df["shrimp_accel_peak"].groupby(gid).transform(
        lambda s: s.rolling(bw, center=True, min_periods=1).sum()) / BURST_WINDOW_S
    print(f"[pics] seuil {thr:.0f} mm/s² | {peak.sum()} pics | "
          f"{peak.sum() / df.groupby('animal_id').ngroups:.1f} par individu en moyenne")

    # ── 4. Sens de nage ───────────────────────────────────────
    #   shrimp_forward_speed : vitesse projetée sur l'axe du corps (mm/s), > 0 avance, < 0 recule
    df["shrimp_forward_speed"] = (vx * sh_x + vy * sh_y) / sh_n * MM * FPS
    df.loc[df["shrimp_speed_mm_s"] == 0, "shrimp_forward_speed"] = 0.0
    df["shrimp_forward_log"] = signed_log(df["shrimp_forward_speed"])
    #   shrimp_swim_dir_deg : angle mouvement / axe du corps, 0 = avant, 180 = arrière (NaN si immobile)
    df["shrimp_swim_dir_deg"] = angle_between_vectors(vx, vy, sh_x, sh_y).where(
        df["shrimp_speed_mm_s"] > SWIM_MIN_SPEED)

    # ── 5. Vitesse angulaire du corps (°/s, sans signe) ───────
    body = np.arctan2(sh_y, sh_x)
    d = (body.groupby(gid).diff() + np.pi) % (2 * np.pi) - np.pi
    d = np.degrees(d.abs().fillna(0))
    d[d < ANGLE_NOISE_DEG] = 0.0
    turn = d.groupby(gid).transform(lambda s: s.rolling(VAR_WINDOW, center=True, min_periods=1).sum()) \
        / (VAR_WINDOW / FPS)
    df["shrimp_turn_speed_deg_s"] = turn.clip(upper=turn.quantile(0.995))
    df["shrimp_turn_log"] = np.log1p(df["shrimp_turn_speed_deg_s"])

    # ── 6. La seiche vue par la crevette + distance ───────────
    #   angle entre l'axe de la crevette (milieu -> tête) et le vecteur crevette -> seiche
    #   0° = la seiche est devant sa tête, 90° = sur le côté, 180° = derrière sa queue
    #   (même définition que attack_side côté seiche)
    cx = (df["Apex_x"] + df["TopMantle_x"]) / 2
    cy = (df["Apex_y"] + df["TopMantle_y"]) / 2
    df["shrimp_view_cuttle"] = angle_between_vectors(sh_x, sh_y, cx - df["MidShrimp_x"], cy - df["MidShrimp_y"])
    df["distance_mm"]  = np.sqrt((df["TopMantle_x"] - df["MidShrimp_x"]) ** 2 +
                                 (df["TopMantle_y"] - df["MidShrimp_y"]) ** 2) * MM
    df["distance_log"] = np.log1p(df["distance_mm"])

    # ── 7. Vérification ────────────────────────────────────────
    features = ["shrimp_speed_log", "shrimp_accel_log", "shrimp_burst_rate", "shrimp_forward_log",
                "shrimp_turn_log", "shrimp_view_cuttle", "distance_log"]
    print("\n=== NaN / Inf / résumé (médiane pre | hunt) ===")
    for f in features + ["shrimp_speed_mm_s", "cuttle_speed_mm_s", "shrimp_swim_dir_deg", "shrimp_turn_speed_deg_s", "distance_mm"]:
        n_nan, n_inf = df[f].isna().sum(), np.isinf(df[f]).sum()
        ok = "✅" if (n_nan == 0 or f == "shrimp_swim_dir_deg") and n_inf == 0 else "❌"
        med = df.groupby("phase")[f].median()
        print(f"  {ok} {f:<26} NaN={n_nan:<7} min={df[f].min():9.2f}  max={df[f].max():9.2f}  "
              f"médiane pre={med.get('pre', np.nan):8.2f} | hunt={med.get('hunt', np.nan):8.2f}")

    corr = df[features].corr().round(2)
    print("\nPaires très corrélées (|r| > 0.7) — redondance possible :")
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            if abs(corr.loc[f1, f2]) > 0.7:
                print(f"  ⚠️  {f1} ↔ {f2} : r={corr.loc[f1, f2]:.2f}")

    # ── 8. Sauvegarde ─────────────────────────────────────────
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n[OK] CSV sauvegardé → {SAVE_PATH}  ({len(df)} frames, {df.shape[1]} colonnes)")
