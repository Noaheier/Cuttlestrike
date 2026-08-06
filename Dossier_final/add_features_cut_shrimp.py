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

# Référentiel du bac — axe de référence fixe (1,1) normalisé
REF_DX = 1 / np.sqrt(2)
REF_DY = 1 / np.sqrt(2)


for SMOOTH_WINDOW in ALL_SMOOTH_WINDOW:

    print(f"\n{'='*60}")
    print(f"SMOOTH_WINDOW = {SMOOTH_WINDOW} frames")
    print(f"{'='*60}")

    SAVE_PATH = (rf"C:\Users\nonoh\Documents\01Travail\02_CREC\cuttle_strike"
                 rf"\cut_shrimp_with_features_smooth_{SMOOTH_WINDOW}.csv")

    # ── 1. Chargement ─────────────────────────────────────────────
    df = pd.read_csv(
        r"E:/01_Travail/CREC/cuttle_strike/combined_df/cut_combined_cuttlefish_tracking.csv",
        on_bad_lines="skip")
    print(f"[OK] {len(df)} frames | {df['animal_id'].nunique()} individus")


    # ── 2. Lissage positions ───────────────────────────────────────
    # Même logique que pour la seiche : rolling centré avec protection
    # des bords pour éviter les artefacts en début/fin de séquence
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

    shrimp_cols = ["HeadShrimp_x", "HeadShrimp_y",
                   "MidShrimp_x",  "MidShrimp_y",
                   "TailShrimp_x", "TailShrimp_y"]

    w = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW + 1
    for col in shrimp_cols:
        if col in df.columns:
            df[col] = df.groupby("animal_id")[col].transform(
                lambda s: smooth_group(s, w))

    print(f"[OK] Lissage appliqué (window={w}) sur {shrimp_cols}")


    # ── 3. Distance seiche–crevette ────────────────────────────────
    # Euclidienne entre TopMantle (seiche) et MidShrimp (crevette)
    # Log-transformée car distribution très asymétrique
    df["shrimp_distance_to_cuttle_raw"] = np.sqrt(
        (df["TopMantle_x"] - df["MidShrimp_x"])**2 +
        (df["TopMantle_y"] - df["MidShrimp_y"])**2)
    df["shrimp_distance_to_cuttle"] = np.log1p(
        df["shrimp_distance_to_cuttle_raw"])

    print(f"[distance] raw max={df['shrimp_distance_to_cuttle_raw'].max():.1f}px"
          f"  → log max={df['shrimp_distance_to_cuttle'].max():.3f}")


    # ── 4. Vitesse et jerk de la crevette ─────────────────────────
    # Centre de masse crevette = moyenne Head + Tail
    # Plus stable que HeadShrimp seul (moins sensible aux erreurs DLC)
    df["shrimp_cx"] = (df["HeadShrimp_x"] + df["TailShrimp_x"]) / 2
    df["shrimp_cy"] = (df["HeadShrimp_y"] + df["TailShrimp_y"]) / 2

    def compute_speed_jerk(group):
        x = group["shrimp_cx"].copy().astype(float)
        y = group["shrimp_cy"].copy().astype(float)

        # Nettoyage sauts aberrants (même seuil que seiche)
        jump = np.sqrt(x.diff()**2 + y.diff()**2)
        x[jump > MAX_JUMP_PX] = np.nan
        y[jump > MAX_JUMP_PX] = np.nan
        x = x.interpolate("linear").ffill().bfill()
        y = y.interpolate("linear").ffill().bfill()

        # Vitesse lissée — rolling centré sur les composantes séparément
        # puis norme → évite les pics artificiels
        vx = x.diff().rolling(SPEED_WINDOW, center=True, min_periods=1).mean()
        vy = y.diff().rolling(SPEED_WINDOW, center=True, min_periods=1).mean()
        speed_smooth = np.sqrt(vx**2 + vy**2)

        # Accélération vectorielle
        accel = np.sqrt(
            vx.diff(ACCEL_WINDOW)**2 + vy.diff(ACCEL_WINDOW)**2
        ) / ACCEL_WINDOW

        # Jerk = variation de l'accélération → détecte les esquives brusques
        # Log-transformé car distribution très asymétrique (pic à 0)
        jerk_raw = accel.diff(JERK_WINDOW).abs()

        group = group.copy()
        group["shrimp_speed_smooth"] = speed_smooth
        group["shrimp_accel"]        = accel
        group["shrimp_jerk_raw"]     = jerk_raw
        return group

    df = df.groupby("animal_id", group_keys=False).apply(compute_speed_jerk)

    # Log-transformation vitesse
    df["shrimp_speed_smooth_raw"] = df["shrimp_speed_smooth"].copy()
    df["shrimp_speed_smooth"]     = np.log1p(df["shrimp_speed_smooth"])

    # Clip accélération
    p975_accel       = df["shrimp_accel"].quantile(0.975)
    df["shrimp_accel"] = df["shrimp_accel"].clip(lower=0, upper=p975_accel)

    # Log + clip jerk
    p95_jerk          = df["shrimp_jerk_raw"].quantile(0.95)
    df["shrimp_jerk"] = np.log1p(
        df["shrimp_jerk_raw"].clip(upper=p95_jerk).fillna(0))

    df[["shrimp_speed_smooth","shrimp_accel","shrimp_jerk"]] = \
        df[["shrimp_speed_smooth","shrimp_accel","shrimp_jerk"]].fillna(0)

    print(f"[speed]  log max={df['shrimp_speed_smooth'].max():.3f}")
    print(f"[accel]  clip [0, {p975_accel:.4f}]")
    print(f"[jerk]   log max={df['shrimp_jerk'].max():.3f}")


    # ── 5. Orientation absolue de la crevette ─────────────────────
    # Axe corps = Tail → Head (direction vers laquelle elle "regarde")
    # Angle par rapport au référentiel fixe (1,1) du bac
    #
    # → Mesure si la crevette tourne en rond dans le bac
    #   ou garde une orientation stable
    #
    # arctan2 donne un angle signé [-180°, +180°]
    # → on prend l'angle absolu par rapport à (1,1) : [0°, 180°]

    body_dx = df["HeadShrimp_x"] - df["TailShrimp_x"]
    body_dy = df["HeadShrimp_y"] - df["TailShrimp_y"]
    body_norm = np.sqrt(body_dx**2 + body_dy**2) + 1e-9

    # cos(angle) entre axe corps et référentiel (1,1)
    # → 1 = alignée avec le bac, 0 = perpendiculaire, -1 = opposée
    cos_ref = (body_dx/body_norm * REF_DX + body_dy/body_norm * REF_DY)
    df["shrimp_orientation_abs"] = np.degrees(np.arccos(np.clip(cos_ref, -1, 1)))

    # Variation d'orientation sur fenêtre glissante
    # = rotation accumulée = "est-ce qu'elle tourne beaucoup ?"
    # Même logique que orientation_variation pour la seiche
    ANGULAR_WINDOW = max(SMOOTH_WINDOW * 4, 20)

    def angular_amplitude_shrimp(group, window):
        # Angle absolu de l'axe corps dans le repère image
        angles = np.arctan2(
            group["HeadShrimp_y"] - group["TailShrimp_y"],
            group["HeadShrimp_x"] - group["TailShrimp_x"])
        # Différences angulaires frame à frame (modulo 2π)
        diffs = ((angles.diff() + np.pi) % (2*np.pi) - np.pi).abs()
        # Somme glissante = rotation totale accumulée sur la fenêtre
        total_rotation = diffs.rolling(window, center=True,
                                        min_periods=1).sum()
        return total_rotation * 180 / np.pi

    df["shrimp_orientation_variation"] = (
        df.groupby("animal_id", group_keys=False)
        .apply(lambda g: angular_amplitude_shrimp(g, ANGULAR_WINDOW))
        .fillna(0))

    p95_ov = df["shrimp_orientation_variation"].quantile(0.95)
    df["shrimp_orientation_variation"] = \
        df["shrimp_orientation_variation"].clip(upper=p95_ov)

    print(f"[orient_variation] window={ANGULAR_WINDOW}  "
          f"p95={p95_ov:.2f}°  max={df['shrimp_orientation_variation'].max():.2f}°")


    # ── 6. Orientation face/dos à la seiche ───────────────────────
    # Angle entre l'axe corps de la crevette (Tail→Head)
    # et le vecteur crevette→seiche (MidShrimp→TopMantle)
    #
    # → 0°   : crevette fait FACE à la seiche (Head pointé vers seiche)
    # → 180° : crevette tourne le DOS à la seiche
    # → 90°  : crevette est perpendiculaire à la seiche
    #
    # Valeur dans [0°, 180°] — symétrique gauche/droite
    # (on ne distingue pas si la seiche est à gauche ou à droite)

    # Axe corps crevette : Tail → Head
    shrimp_dx = df["HeadShrimp_x"] - df["TailShrimp_x"]
    shrimp_dy = df["HeadShrimp_y"] - df["TailShrimp_y"]

    # Vecteur crevette → seiche
    to_cuttle_dx = df["TopMantle_x"] - df["MidShrimp_x"]
    to_cuttle_dy = df["TopMantle_y"] - df["MidShrimp_y"]

    # Produit scalaire normalisé → arccos → angle [0°, 180°]
    shrimp_norm    = np.sqrt(shrimp_dx**2    + shrimp_dy**2)    + 1e-9
    to_cuttle_norm = np.sqrt(to_cuttle_dx**2 + to_cuttle_dy**2) + 1e-9

    cos_face = ((shrimp_dx/shrimp_norm) * (to_cuttle_dx/to_cuttle_norm) +
                (shrimp_dy/shrimp_norm) * (to_cuttle_dy/to_cuttle_norm))

    df["shrimp_facing_cuttle"] = np.degrees(np.arccos(np.clip(cos_face, -1, 1)))

    # Lissage léger — réduire le bruit de détection DLC
    df["shrimp_facing_cuttle"] = (
        df.groupby("animal_id")["shrimp_facing_cuttle"]
        .transform(lambda g: g.rolling(SPEED_WINDOW, center=True,
                                        min_periods=1).mean())
        .fillna(90))  # 90° = neutre si NaN

    print(f"[facing_cuttle] mean={df['shrimp_facing_cuttle'].mean():.1f}°  "
          f"(0°=face, 180°=dos)  std={df['shrimp_facing_cuttle'].std():.1f}°")


    # ── 7. Features finales ────────────────────────────────────────
    shrimp_features = [
        "shrimp_distance_to_cuttle",     # distance log → position relative
        "shrimp_speed_smooth",            # vitesse translation (log)
        "shrimp_accel",                   # changement de vitesse vectoriel
        "shrimp_jerk",                    # esquives brusques (log)
        "shrimp_orientation_abs",         # orientation absolue dans le bac
        "shrimp_orientation_variation",   # rotation accumulée → tourne vs va droit
        "shrimp_facing_cuttle",           # face(0°) ou dos(180°) à la seiche
    ]
    shrimp_features = [f for f in shrimp_features if f in df.columns]


    # ── 8. Vérification finale ─────────────────────────────────────
    print(f"\n=== NaN / Inf ===")
    for feat in shrimp_features:
        n_nan = df[feat].isna().sum()
        n_inf = np.isinf(df[feat]).sum()
        ok    = "✅" if n_nan == 0 and n_inf == 0 else "❌"
        print(f"  {ok} {feat:<35} NaN={n_nan}  Inf={n_inf}  "
              f"min={df[feat].min():.3f}  max={df[feat].max():.3f}  "
              f"mean={df[feat].mean():.3f}")

    print(f"\n=== % de zéros ===")
    for feat in shrimp_features:
        print(f"  {feat:<35} {(df[feat]==0).mean()*100:.1f}%")

    print(f"\n=== Corrélations entre features ===")
    corr_mat = df[shrimp_features].corr().round(2)
    print(corr_mat)
    print("\nPaires très corrélées (|r| > 0.7) :")
    for i, f1 in enumerate(shrimp_features):
        for f2 in shrimp_features[i+1:]:
            r = corr_mat.loc[f1, f2]
            if abs(r) > 0.7:
                print(f"  ⚠️  {f1} ↔ {f2} : r={r:.2f}")


    # ── 9. Sauvegarde ─────────────────────────────────────────────
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n[OK] CSV sauvegardé → {SAVE_PATH}")
    print(f"     Features crevette : {shrimp_features}")