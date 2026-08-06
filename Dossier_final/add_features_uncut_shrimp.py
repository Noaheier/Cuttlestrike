import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════
# PARAMÈTRES
# ══════════════════════════════════════════════════════════════
ALL_SMOOTH_WINDOW = (1, 5, 10, 20, 50)
SPEED_WINDOW      = 10
ACCEL_WINDOW      = 5
JERK_WINDOW       = 3
MAX_JUMP_PX       = 50
CENTER_RADIUS     = 30    # px — rayon filtre artefacts DLC centre image

REF_DX = 1 / np.sqrt(2)
REF_DY = 1 / np.sqrt(2)

# ── Chargement unique hors boucle — même données pour tous les smooth ──
print("[LOAD] Chargement du dataset non-cropé...")
df_raw = pd.read_csv(
    r"E:/01_Travail/CREC/cuttle_strike/combined_df/combined_cuttlefish_tracking.csv",
    on_bad_lines="skip")
print(f"[OK] {len(df_raw)} frames | {df_raw['animal_id'].nunique()} individus")

# ── Chargement Classeur1 — frame de début de capture ──────────
df_start = pd.read_excel(
    r"E:\01_Travail\CREC\First_Capture\analyse\Classeur1.xlsx")
df_start = df_start[["video","start"]].dropna(subset=["start"])
df_start["video"] = df_start["video"].astype(int)
df_start["start"] = df_start["start"].astype(int)
print(f"[OK] Classeur1 : {len(df_start)} individus avec frame start")
print(f"     start : min={df_start['start'].min()} "
      f"max={df_start['start'].max()} "
      f"mean={df_start['start'].mean():.0f}")

# ── Merge — ajouter start à chaque frame ──────────────────────
# On joint sur animal_id == video
df_raw = df_raw.merge(
    df_start.rename(columns={"video": "animal_id", "start": "capture_start"}),
    on="animal_id", how="left")

n_with_start = df_raw["capture_start"].notna().sum()
n_individus_ok = df_raw[df_raw["capture_start"].notna()]["animal_id"].nunique()
print(f"[OK] {n_individus_ok} individus avec capture_start connu")
print(f"     {df_raw['animal_id'].nunique() - n_individus_ok} individus "
      f"sans capture_start → exclus")

# ── Filtre : garder uniquement les frames PRÉ-CAPTURE ─────────
# frame < capture_start = avant le début de la phase de capture
# C'est là qu'on cherche ce qui influence le choix de prédation
df_raw = df_raw[
    df_raw["capture_start"].notna() &
    (df_raw["frame"] < df_raw["capture_start"])
].copy()

print(f"\n[PRÉ-CAPTURE] {len(df_raw)} frames retenues")
print(f"  {df_raw['animal_id'].nunique()} individus")
print(f"  Durée moy. pré-capture : "
      f"{df_raw.groupby('animal_id').size().mean():.0f} frames/individu")
print(f"  Durée min : {df_raw.groupby('animal_id').size().min()} frames")
print(f"  Durée max : {df_raw.groupby('animal_id').size().max()} frames")

# Vérification — quelques individus
print("\n  Aperçu par individu :")
for aid in df_raw["animal_id"].unique()[:5]:
    sub = df_raw[df_raw["animal_id"] == aid]
    start_val = sub["capture_start"].iloc[0]
    print(f"    ID={aid:>4}  frames=[{sub['frame'].min()}, "
          f"{sub['frame'].max()}]  "
          f"start={start_val}  "
          f"n_frames={len(sub)}")


# ══════════════════════════════════════════════════════════════
# BOUCLE SMOOTH WINDOWS
# ══════════════════════════════════════════════════════════════
for SMOOTH_WINDOW in ALL_SMOOTH_WINDOW:

    print(f"\n{'='*60}")
    print(f"SMOOTH_WINDOW = {SMOOTH_WINDOW} frames")
    print(f"{'='*60}")

    SAVE_PATH = (
        rf"C:\Users\nonoh\Documents\01Travail\02_CREC\cuttle_strike"
        rf"\shrimp_precapture_features_smooth_{SMOOTH_WINDOW}.csv")

    df = df_raw.copy()

    # ── 1.5 Nettoyage artefacts DLC AVANT lissage ─────────────────
    # Deux types d'artefacts :
    #   1. Point mis au centre de l'image quand le tracker perd la cible
    #   2. Sauts aberrants entre frames consécutives
    # On remplace par NaN puis interpolation linéaire
    # → le lissage qui suit travaille sur des données propres

    shrimp_cols = ["HeadShrimp_x", "HeadShrimp_y",
                   "MidShrimp_x",  "MidShrimp_y",
                   "TailShrimp_x", "TailShrimp_y"]

    # Estimation du centre de l'image depuis les données
    cx_est = df["HeadShrimp_x"].quantile(0.50)
    cy_est = df["HeadShrimp_y"].quantile(0.50)
    print(f"[CLEAN] Centre image estimé : ({cx_est:.0f}, {cy_est:.0f})")

    def clean_points(group):
        """
        Nettoie les artefacts DLC pour un individu.
        Appliqué colonne par colonne sur les paires (x,y).
        """
        g = group.copy()
        for cx_col, cy_col in [("HeadShrimp_x","HeadShrimp_y"),
                                ("MidShrimp_x", "MidShrimp_y"),
                                ("TailShrimp_x","TailShrimp_y")]:
            x = g[cx_col].astype(float)
            y = g[cy_col].astype(float)

            # Filtre 1 — points au centre image (tracker perdu)
            mask_center = ((np.abs(x - cx_est) < CENTER_RADIUS) &
                           (np.abs(y - cy_est) < CENTER_RADIUS))
            x[mask_center] = np.nan
            y[mask_center] = np.nan

            # Filtre 2 — sauts aberrants frame à frame
            dist = np.sqrt(x.diff()**2 + y.diff()**2)
            x[dist > MAX_JUMP_PX] = np.nan
            y[dist > MAX_JUMP_PX] = np.nan

            # Interpolation + remplissage bords
            x = x.interpolate("linear").ffill().bfill()
            y = y.interpolate("linear").ffill().bfill()

            g[cx_col] = x
            g[cy_col] = y
        return g

    df = df.groupby("animal_id", group_keys=False).apply(clean_points)
    print(f"[OK] Nettoyage DLC terminé")


    # ── 2. Lissage positions ───────────────────────────────────────
    def smooth_group(series, window):
        half   = window // 2
        n      = len(series)
        result = series.rolling(window, center=True, min_periods=1).mean()
        if half > 0 and n > 2 * half:
            result.iloc[:half]  = series.iloc[:half]
            result.iloc[-half:] = series.iloc[-half:]
        elif n <= 2 * half:
            result = series.copy()
        return result

    w = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW + 1
    for col in shrimp_cols:
        if col in df.columns:
            df[col] = df.groupby("animal_id")[col].transform(
                lambda s: smooth_group(s, w))
    print(f"[OK] Lissage (window={w})")


    # ── 3. Distance seiche–crevette ────────────────────────────────
    df["shrimp_distance_to_cuttle_raw"] = np.sqrt(
        (df["TopMantle_x"] - df["MidShrimp_x"])**2 +
        (df["TopMantle_y"] - df["MidShrimp_y"])**2)
    df["shrimp_distance_to_cuttle"] = np.log1p(
        df["shrimp_distance_to_cuttle_raw"])
    print(f"[distance] raw max={df['shrimp_distance_to_cuttle_raw'].max():.1f}px"
          f"  → log max={df['shrimp_distance_to_cuttle'].max():.3f}")


    # ── 4. Temps relatif avant capture ────────────────────────────
    # Feature clé pour la phase pré-capture :
    # t=0 = début de la vidéo, t=1 = frame de capture
    # Permet de voir si les comportements changent à l'approche
    # de la décision de prédation
    df["time_to_capture"] = (
        df["capture_start"] - df["frame"]
    ).astype(float)
    # Aussi en version normalisée [0,1] par individu
    df["time_norm"] = df.groupby("animal_id")["time_to_capture"].transform(
        lambda g: 1 - (g / g.max()))   # 0=début vidéo, 1=juste avant capture
    print(f"[time_to_capture] max={df['time_to_capture'].max():.0f} frames")
    print(f"[time_norm]       range=[0, 1] par individu")


    # ── 5. Vitesse et jerk ────────────────────────────────────────
    df["shrimp_cx"] = (df["HeadShrimp_x"] + df["TailShrimp_x"]) / 2
    df["shrimp_cy"] = (df["HeadShrimp_y"] + df["TailShrimp_y"]) / 2

    def compute_speed_jerk(group):
        x = group["shrimp_cx"].copy().astype(float)
        y = group["shrimp_cy"].copy().astype(float)

        vx = x.diff().rolling(SPEED_WINDOW, center=True, min_periods=1).mean()
        vy = y.diff().rolling(SPEED_WINDOW, center=True, min_periods=1).mean()
        speed_smooth = np.sqrt(vx**2 + vy**2)

        accel = np.sqrt(
            vx.diff(ACCEL_WINDOW)**2 + vy.diff(ACCEL_WINDOW)**2
        ) / ACCEL_WINDOW

        jerk_raw = accel.diff(JERK_WINDOW).abs()

        group = group.copy()
        group["shrimp_speed_smooth"] = speed_smooth
        group["shrimp_accel"]        = accel
        group["shrimp_jerk_raw"]     = jerk_raw
        return group

    df = df.groupby("animal_id", group_keys=False).apply(compute_speed_jerk)

    df["shrimp_speed_smooth_raw"] = df["shrimp_speed_smooth"].copy()
    df["shrimp_speed_smooth"]     = np.log1p(df["shrimp_speed_smooth"])

    p975_accel         = df["shrimp_accel"].quantile(0.975)
    df["shrimp_accel"] = df["shrimp_accel"].clip(lower=0, upper=p975_accel)

    p95_jerk          = df["shrimp_jerk_raw"].quantile(0.95)
    df["shrimp_jerk"] = np.log1p(
        df["shrimp_jerk_raw"].clip(upper=p95_jerk).fillna(0))

    df[["shrimp_speed_smooth","shrimp_accel","shrimp_jerk"]] = \
        df[["shrimp_speed_smooth","shrimp_accel","shrimp_jerk"]].fillna(0)

    print(f"[speed] log max={df['shrimp_speed_smooth'].max():.3f}")
    print(f"[accel] clip [0, {p975_accel:.4f}]")
    print(f"[jerk]  log max={df['shrimp_jerk'].max():.3f}")


    # ── 6. Orientation absolue dans le bac ────────────────────────
    body_dx   = df["HeadShrimp_x"] - df["TailShrimp_x"]
    body_dy   = df["HeadShrimp_y"] - df["TailShrimp_y"]
    body_norm = np.sqrt(body_dx**2 + body_dy**2) + 1e-9
    cos_ref   = (body_dx/body_norm * REF_DX + body_dy/body_norm * REF_DY)
    df["shrimp_orientation_abs"] = np.degrees(
        np.arccos(np.clip(cos_ref, -1, 1)))

    ANGULAR_WINDOW = max(SMOOTH_WINDOW * 4, 20)

    def angular_amplitude_shrimp(group, window):
        angles = np.arctan2(
            group["HeadShrimp_y"] - group["TailShrimp_y"],
            group["HeadShrimp_x"] - group["TailShrimp_x"])
        diffs  = ((angles.diff() + np.pi) % (2*np.pi) - np.pi).abs()
        return diffs.rolling(window, center=True, min_periods=1).sum() * 180/np.pi

    df["shrimp_orientation_variation"] = (
        df.groupby("animal_id", group_keys=False)
        .apply(lambda g: angular_amplitude_shrimp(g, ANGULAR_WINDOW))
        .fillna(0))
    p95_ov = df["shrimp_orientation_variation"].quantile(0.95)
    df["shrimp_orientation_variation"] = \
        df["shrimp_orientation_variation"].clip(upper=p95_ov)
    print(f"[orient_var] window={ANGULAR_WINDOW}  p95={p95_ov:.2f}°")


    # ── 7. Orientation face/dos à la seiche ───────────────────────
    shrimp_dx      = df["HeadShrimp_x"] - df["TailShrimp_x"]
    shrimp_dy      = df["HeadShrimp_y"] - df["TailShrimp_y"]
    to_cuttle_dx   = df["TopMantle_x"]  - df["MidShrimp_x"]
    to_cuttle_dy   = df["TopMantle_y"]  - df["MidShrimp_y"]
    shrimp_norm    = np.sqrt(shrimp_dx**2    + shrimp_dy**2)    + 1e-9
    to_cuttle_norm = np.sqrt(to_cuttle_dx**2 + to_cuttle_dy**2) + 1e-9
    cos_face       = ((shrimp_dx/shrimp_norm) * (to_cuttle_dx/to_cuttle_norm) +
                      (shrimp_dy/shrimp_norm) * (to_cuttle_dy/to_cuttle_norm))
    df["shrimp_facing_cuttle"] = np.degrees(np.arccos(np.clip(cos_face,-1,1)))
    df["shrimp_facing_cuttle"] = (
        df.groupby("animal_id")["shrimp_facing_cuttle"]
        .transform(lambda g: g.rolling(SPEED_WINDOW,center=True,min_periods=1).mean())
        .fillna(90))
    print(f"[facing] mean={df['shrimp_facing_cuttle'].mean():.1f}°  "
          f"std={df['shrimp_facing_cuttle'].std():.1f}°")


    # ── 8. Distance au bord du bac ────────────────────────────────
    # Feature spécifique pré-capture : la crevette est-elle
    # près du bord (comportement de fuite) ou au centre ?
    # On estime le centre et le rayon du bac depuis les données
    x_all  = df["HeadShrimp_x"].quantile([0.02, 0.98])
    y_all  = df["HeadShrimp_y"].quantile([0.02, 0.98])
    cx_bac = (x_all.iloc[0] + x_all.iloc[1]) / 2
    cy_bac = (y_all.iloc[0] + y_all.iloc[1]) / 2
    r_bac  = min(x_all.iloc[1]-x_all.iloc[0],
                 y_all.iloc[1]-y_all.iloc[0]) / 2

    df["shrimp_dist_to_center"] = np.sqrt(
        (df["HeadShrimp_x"] - cx_bac)**2 +
        (df["HeadShrimp_y"] - cy_bac)**2)
    # Normalisé par le rayon du bac → [0=centre, 1=bord, >1=hors bac]
    df["shrimp_dist_to_center_norm"] = df["shrimp_dist_to_center"] / r_bac
    print(f"[dist_center] centre=({cx_bac:.0f},{cy_bac:.0f})  "
          f"rayon={r_bac:.0f}px  "
          f"mean_norm={df['shrimp_dist_to_center_norm'].mean():.2f}")


    # ── 9. Features finales ───────────────────────────────────────
    shrimp_features = [
        "shrimp_distance_to_cuttle",     # distance log à la seiche
        "shrimp_speed_smooth",            # vitesse (log)
        "shrimp_accel",                   # accélération vectorielle
        "shrimp_jerk",                    # esquives brusques (log)
        "shrimp_orientation_abs",         # orientation absolue dans le bac
        "shrimp_orientation_variation",   # rotation accumulée
        "shrimp_facing_cuttle",           # face(0°) ou dos(180°) à la seiche
        "shrimp_dist_to_center_norm",     # proximité bord du bac [0-1]
        "time_to_capture",                # frames restantes avant capture
        "time_norm",                      # position temporelle normalisée [0-1]
    ]
    shrimp_features = [f for f in shrimp_features if f in df.columns]


    # ── 10. Vérification ──────────────────────────────────────────
    print(f"\n=== NaN / Inf ===")
    for feat in shrimp_features:
        n_nan = df[feat].isna().sum()
        n_inf = np.isinf(df[feat]).sum()
        ok    = "✅" if n_nan == 0 and n_inf == 0 else "❌"
        print(f"  {ok} {feat:<35} NaN={n_nan}  Inf={n_inf}  "
              f"min={df[feat].min():.3f}  max={df[feat].max():.3f}  "
              f"mean={df[feat].mean():.3f}")

    print(f"\n=== Corrélations ===")
    corr_mat = df[shrimp_features].corr().round(2)
    print(corr_mat)
    print("\nPaires très corrélées (|r| > 0.7) :")
    for i, f1 in enumerate(shrimp_features):
        for f2 in shrimp_features[i+1:]:
            r = corr_mat.loc[f1, f2]
            if abs(r) > 0.7:
                print(f"  ⚠️  {f1} ↔ {f2} : r={r:.2f}")


    # ── 11. Sauvegarde ────────────────────────────────────────────
    # On garde aussi capture_start et time_to_capture dans le CSV
    # pour pouvoir reconstruire la temporalité plus tard
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n[OK] → {SAVE_PATH}")
    print(f"     {len(df)} frames pré-capture | "
          f"{df['animal_id'].nunique()} individus")
    print(f"     Features : {shrimp_features}")