import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════
# PARAMÈTRES — change uniquement ici
# ══════════════════════════════════════════════════════════════
ALL_SMOOTH_WINDOW  = (1,5,10,20,50)
SPEED_WINDOW   = 10
ACCEL_WINDOW   = 5
MAX_JUMP_PX    = 50
ANGLE_SHIFT    = 5
JERK_WINDOW = 3


def par_animal(df, func, **kw):
    """
    Applique func à chaque individu puis réassemble.
    En pandas 3.x, groupby(...).apply(...) exclut la colonne de groupage ;
    l'itération sur un groupby la conserve, quelle que soit la version.
    """
    out = [func(g, **kw) for _, g in df.groupby("animal_id", sort=False)]
    return pd.concat(out).sort_index()


for SMOOTH_WINDOW in ALL_SMOOTH_WINDOW:

    print(f"\n=== SMOOTH_WINDOW={SMOOTH_WINDOW} frames | ")
    SAVE_PATH = rf"C:\Users\nonoh\Documents\01Travail\02_CREC\cuttle_strike\cut_cuttlefish_with_features_smooth_{SMOOTH_WINDOW}.csv"
    # ══════════════════════════════════════════════════════════════


    # ── 1. Chargement ─────────────────────────────────────────────
    df   = pd.read_csv(r"E:/01_Travail/CREC/cuttle_strike/combined_df/cut_combined_cuttlefish_tracking.csv",
                    on_bad_lines='skip')
    Data = pd.read_excel(r"E:/01_Travail/CREC/First_Capture/analyse/Data_cuttlfish_strike.xlsx")
    df = df.reset_index(drop=True)   # par_animal réassemble via sort_index()
    print(f"[OK] Chargé : {len(df)} frames | {df['animal_id'].nunique()} individus")


    # ── 2. Lissage DLC avec protection des bords ──────────────────
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

    def smooth_df_dlc(df, window):
        w       = window if window % 2 == 1 else window + 1
        exclude = {"animal_id", "frame"}
        df_s    = df.copy()
        cols    = [col for col in df.columns
                if col not in exclude
                and col != df.columns[0]
                and "likelihood" not in str(col).lower()
                and not isinstance(col, tuple)]
        for col in cols:
            df_s[col] = df.groupby("animal_id")[col].transform(
                lambda s: smooth_group(s, w)
            )
        return df_s

    df = smooth_df_dlc(df, window=SMOOTH_WINDOW)


    # ── 3. Distance seiche–crevette ───────────────────────────────
    # Log-transformée car distribution très asymétrique (longue queue)
    # log1p(x) = log(x+1) → garde les zéros à 0, compresse les extrêmes

    df["distance_to_shrimp_raw"] = np.sqrt(
        (df["TopMantle_x"] - df["MidShrimp_x"])**2 +
        (df["TopMantle_y"] - df["MidShrimp_y"])**2
    )
    df["distance_to_shrimp"] = np.log1p(df["distance_to_shrimp_raw"])

    print(f"[distance_to_shrimp] log-transformé  "
        f"raw max={df['distance_to_shrimp_raw'].max():.1f}px  "
        f"→ log max={df['distance_to_shrimp'].max():.3f}")


    # ── 4. Vitesse et accélération ────────────────────────────────
    def clean_speed_accel_v1(group, x_col, y_col,
                            max_jump_px, speed_window, accel_window):
        x = group[x_col].copy().astype(float)
        y = group[y_col].copy().astype(float)

        jump = np.sqrt(x.diff()**2 + y.diff()**2)
        x[jump > max_jump_px] = np.nan
        y[jump > max_jump_px] = np.nan
        x = x.interpolate("linear").ffill().bfill()
        y = y.interpolate("linear").ffill().bfill()

        vx = x.diff().rolling(speed_window, center=True, min_periods=1).mean()
        vy = y.diff().rolling(speed_window, center=True, min_periods=1).mean()
        speed_smooth = np.sqrt(vx**2 + vy**2)

        # Accélération vectorielle — capture les virages
        # (dvx, dvy séparément puis norme, toujours ≥ 0)
        accel_vec = np.sqrt(
            vx.diff(accel_window)**2 + vy.diff(accel_window)**2
        ) / accel_window

        group = group.copy()
        group["_speed_raw"]    = np.sqrt(x.diff()**2 + y.diff()**2)
        group["_speed_smooth"] = speed_smooth
        group["_accel"]        = accel_vec
        return group


    # Centre de masse seiche : insensible à la rotation sur place
    # Rotation → Apex et TopMantle bougent en sens inverse → moyenne stable
    # Translation → les deux bougent ensemble → speed élevée
    df["body_cx"] = (df["Apex_x"] + df["TopMantle_x"]) / 2
    df["body_cy"] = (df["Apex_y"] + df["TopMantle_y"]) / 2

    # Seiche — vitesse sur le centre de masse
    df = par_animal(df, clean_speed_accel_v1,
                    x_col="body_cx", y_col="body_cy",
                    max_jump_px=MAX_JUMP_PX,
                    speed_window=SPEED_WINDOW, accel_window=ACCEL_WINDOW)
    df["speed_cuttle"]        = df["_speed_raw"]
    df["speed_cuttle_smooth"] = df["_speed_smooth"]
    df["accel_cuttle"]        = df["_accel"]
    df.drop(columns=["_speed_raw", "_speed_smooth", "_accel"], inplace=True)

    # Crevette — pas de rotation propre, HeadShrimp est correct
    df = par_animal(df, clean_speed_accel_v1,
                    x_col="HeadShrimp_x", y_col="HeadShrimp_y",
                    max_jump_px=MAX_JUMP_PX,
                    speed_window=SPEED_WINDOW, accel_window=ACCEL_WINDOW)
    df["speed_shrimp"]        = df["_speed_raw"]
    df["speed_shrimp_smooth"] = df["_speed_smooth"]
    df["accel_shrimp"]        = df["_accel"]
    df.drop(columns=["_speed_raw", "_speed_smooth", "_accel"], inplace=True)

    df[["speed_cuttle","speed_cuttle_smooth","accel_cuttle",
        "speed_shrimp","speed_shrimp_smooth","accel_shrimp"]] = \
    df[["speed_cuttle","speed_cuttle_smooth","accel_cuttle",
        "speed_shrimp","speed_shrimp_smooth","accel_shrimp"]].fillna(0)

    df["speed_difference"] = df["speed_cuttle_smooth"] - df["speed_shrimp_smooth"]

    # Log-transformation vitesse (distribution asymétrique, pic à 0)
    df["speed_cuttle_smooth_raw"] = df["speed_cuttle_smooth"].copy()
    df["speed_cuttle_smooth"]     = np.log1p(df["speed_cuttle_smooth"])
    print(f"[speed_cuttle_smooth] log  "
        f"raw max={df['speed_cuttle_smooth_raw'].max():.2f} "
        f"→ log max={df['speed_cuttle_smooth'].max():.3f}")

    # Clip accélération — toujours ≥ 0 (norme), clip seulement le haut
    p97_5 = df["accel_cuttle"].quantile(0.975)
    df["accel_cuttle"] = df["accel_cuttle"].clip(lower=0, upper=p97_5)
    print(f"[accel_cuttle] clip [0, {p97_5:.4f}]  "
        f"max après={df['accel_cuttle'].max():.4f}")


    # ── 5. Angles ─────────────────────────────────────────────────
    def angle_between_vectors(dx1, dy1, dx2, dy2):
        dot  = dx1*dx2 + dy1*dy2
        norm = np.sqrt(dx1**2+dy1**2) * np.sqrt(dx2**2+dy2**2)
        return np.degrees(np.arccos(np.clip(dot / (norm + 1e-9), -1.0, 1.0)))

    def signed_angle(dx1, dy1, dx2, dy2):
        return np.degrees(np.arctan2(dx1*dy2 - dy1*dx2, dx1*dx2 + dy1*dy2))

    # approach_angle : angle entre axe corps (Apex→TopMantle) et axe crevette
    # PAS de double lissage — les positions sont déjà lissées sur SMOOTH_WINDOW
    df["approach_angle"] = angle_between_vectors(
        df["TopMantle_x"] - df["Apex_x"],       df["TopMantle_y"] - df["Apex_y"],
        df["HeadShrimp_x"] - df["MidShrimp_x"], df["HeadShrimp_y"] - df["MidShrimp_y"],
    )
    print(f"[approach_angle] mean={df['approach_angle'].mean():.1f}°  "
        f"std={df['approach_angle'].std():.1f}°")


    # ── 6. orientation_variation (amplitude angulaire du corps) ───
    # Mesure la ROTATION TOTALE accumulée sur une fenêtre glissante
    # → discrimine "tourne beaucoup" vs "va tout droit"
    # Cohérent avec ANGLE_SHIFT pour la fenêtre
    ANGULAR_WINDOW = max(ANGLE_SHIFT * 4, 20)   # fenêtre = 4× le shift, min 20

    def angular_amplitude(group, window):
        angles = np.arctan2(
            group["TopMantle_y"] - group["Apex_y"],
            group["TopMantle_x"] - group["Apex_x"]
        )
        diffs          = ((angles.diff() + np.pi) % (2*np.pi) - np.pi).abs()
        total_rotation = diffs.rolling(window, center=True, min_periods=1).sum()
        return total_rotation * 180 / np.pi

    df["orientation_variation"] = par_animal(df, angular_amplitude,
                                             window=ANGULAR_WINDOW).fillna(0)
    p95_ov = df["orientation_variation"].quantile(0.95)
    df["orientation_variation"] = df["orientation_variation"].clip(upper=p95_ov)
    print(f"[orientation_variation] window={ANGULAR_WINDOW}  "
        f"p95={p95_ov:.2f}°  max={df['orientation_variation'].max():.2f}°")

    # Vérification orthogonalité speed / orientation_variation
    # Doivent être peu corrélés car ils mesurent des choses différentes
    corr = df["speed_cuttle_smooth"].corr(df["orientation_variation"])
    print(f"[CHECK] corr(speed, orientation_variation) = {corr:.3f}  "
        f"{'✅' if abs(corr) < 0.3 else '⚠️ encore corrélées'}")


    # ── 7. Head turn — rotation de la tête par rapport au corps ───
    # Angle SIGNÉ entre :
    #   axe corps   : Apex → TopMantle
    #   axe tête    : Leye → Reye  (axe inter-oculaire)
    #
    # Interprétation :
    #   ≈ 0°  : tête alignée avec le corps (regard frontal)
    #   > 0°  : tête tournée à droite par rapport au corps
    #   < 0°  : tête tournée à gauche
    #
    # Pertinent pour la seiche : elle peut orienter ses yeux
    # indépendamment du corps pour suivre une cible
    eye_cols_needed = ["Leye_x", "Leye_y", "Reye_x", "Reye_y"]
    missing_eye_head = [c for c in eye_cols_needed if c not in df.columns]

    if missing_eye_head:
        print(f"[WARN] head_turn : colonnes manquantes {missing_eye_head}")
        print(f"       Colonnes 'eye' disponibles : "
            f"{[c for c in df.columns if 'eye' in c.lower()]}")
        df["head_turn"] = 0.0
    else:
        df["head_turn"] = signed_angle(
            # axe corps : Apex → TopMantle
            df["TopMantle_x"] - df["Apex_x"],
            df["TopMantle_y"] - df["Apex_y"],
            # axe tête  : Leye → Reye
            df["Reye_x"] - df["Leye_x"],
            df["Reye_y"] - df["Leye_y"],
        )
        # Lissage léger pour réduire le bruit de détection des yeux
        df["head_turn"] = (
            df.groupby("animal_id")["head_turn"]
            .transform(lambda g: g.rolling(SPEED_WINDOW, center=True,
                                            min_periods=1).mean())
        )
        # Clip symétrique (angle signé → les deux queues)
        p2_5_ht  = df["head_turn"].quantile(0.025)
        p97_5_ht = df["head_turn"].quantile(0.975)
        df["head_turn"] = df["head_turn"].clip(lower=p2_5_ht, upper=p97_5_ht)
        print(f"[head_turn] signé  "
            f"mean={df['head_turn'].mean():.2f}°  "
            f"std={df['head_turn'].std():.2f}°  "
            f"clip [{p2_5_ht:.1f}°, {p97_5_ht:.1f}°]")


    # ── 8. Convergence des yeux ───────────────────────────────────
    eye_cols_conv = ["Leye_x", "Leye_y", "Reye_x", "Reye_y",
                    "LeyeBack_x", "LeyeBack_y", "ReyeBack_x", "ReyeBack_y"]
    missing_eye_conv = [c for c in eye_cols_conv if c not in df.columns]

    if missing_eye_conv:
        print(f"[WARN] eye_convergence : colonnes manquantes {missing_eye_conv}")
        df["eye_convergence"] = 0.0
    else:
        D1 = np.sqrt((df["Leye_x"]     - df["Reye_x"]    )**2 +
                    (df["Leye_y"]     - df["Reye_y"]    )**2)
        D2 = np.sqrt((df["LeyeBack_x"] - df["ReyeBack_x"])**2 +
                    (df["LeyeBack_y"] - df["ReyeBack_y"])**2)

        # transform sur un SeriesGroupBy : non affecté par le changement pandas 3.x
        df["eye_convergence"] = D1 - D2
        df["eye_convergence"] = df.groupby("animal_id")["eye_convergence"].transform(
            lambda g: g.rolling(max(SMOOTH_WINDOW, 3), center=True,
                                min_periods=1).mean())
        p2_5_eye  = df["eye_convergence"].quantile(0.025)
        p97_5_eye = df["eye_convergence"].quantile(0.975)
        df["eye_convergence"] = df["eye_convergence"].clip(lower=p2_5_eye,
                                                            upper=p97_5_eye)
        print(f"[eye_convergence] mean={df['eye_convergence'].mean():.2f}px  "
            f"clip [{p2_5_eye:.2f}, {p97_5_eye:.2f}]")


    # ── 9. Jerk ───────────────────────────────────────────────────
    jerk_raw          = (df.groupby("animal_id")["accel_cuttle"]
                        .transform(lambda g: g.diff(JERK_WINDOW).abs())
                        .fillna(0))
    df["jerk_cuttle"] = np.log1p(jerk_raw.clip(upper=jerk_raw.quantile(0.95)))
    print(f"[jerk_cuttle] mean={df['jerk_cuttle'].mean():.3f}  "
        f"max={df['jerk_cuttle'].max():.3f}")


    # ── 10. Features finales ──────────────────────────────────────
    features = [
        "distance_to_shrimp",     # distance log → position relative
        "approach_angle",          # orientation corps vs crevette
        "orientation_variation",   # rotation accumulée → tourne vs va droit
        "speed_cuttle_smooth",     # vitesse translation (log, centre de masse)
        "accel_cuttle",            # changement de vitesse vectoriel
        "jerk_cuttle",             # transitions brusques (frappe, esquive)
        "head_turn",               # rotation tête par rapport au corps
        "eye_convergence",         # convergence oculaire (focus sur cible)
    ]
    features = [f for f in features if f in df.columns]


    # ── 11. Vérification finale ───────────────────────────────────
    print("\n=== NaN / Inf ===")
    for f in features:
        n_nan = df[f].isna().sum()
        n_inf = np.isinf(df[f]).sum()
        ok    = "✅" if n_nan == 0 and n_inf == 0 else "❌"
        print(f"  {ok} {f:<30} NaN={n_nan}  Inf={n_inf}  "
            f"min={df[f].min():.3f}  max={df[f].max():.3f}  "
            f"mean={df[f].mean():.3f}")

    print("\n=== % de zéros ===")
    for f in features:
        print(f"  {f:<30} {(df[f]==0).mean()*100:.1f}%")

    print("\n=== Corrélations entre features ===")
    corr_mat = df[features].corr().round(2)
    print(corr_mat)
    print("\nPaires très corrélées (|r| > 0.7) — redondance possible :")
    for i, f1 in enumerate(features):
        for f2 in features[i+1:]:
            r = corr_mat.loc[f1, f2]
            if abs(r) > 0.7:
                print(f"  ⚠️  {f1} ↔ {f2} : r={r:.2f}")
                
    # ── 9. Sauvegarde ─────────────────────────────────────────────
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n[OK] CSV sauvegardé → {SAVE_PATH}")
    print(f"     Features finales : {features}")
    # ── 9. Hit point ──────────────────────────────────────────────
    df["animal_id"] = df["animal_id"].astype(str)
    Data["num"]     = Data["num"].astype(str)
    df["frame"]     = df["frame"].astype(int)

    Data["distance_intersection_normalized"] = np.nan
    ok_count = no_match_count = parallel_count = nan_count = 0

    for idx, row in Data.iterrows():
        aid       = row["num"]
        df_animal = df[df["animal_id"] == aid]
        if df_animal.empty:
            no_match_count += 1; continue

        catch_frame = df_animal["frame"].max()
        mask        = (df["animal_id"] == aid) & (df["frame"] == catch_frame)
        if not mask.any():
            no_match_count += 1; continue

        f = df.loc[mask].iloc[0]

        x1, y1 = f["HeadShrimp_x"],    f["HeadShrimp_y"]
        x2, y2 = f["MidShrimp_x"],     f["MidShrimp_y"]
        x3, y3 = f["MidCuttleHead_x"], f["MidCuttleHead_y"]
        x4      = (f["Ltentacles_x"] + f["Rtentacles_x"]) / 2
        y4      = (f["Ltentacles_y"] + f["Rtentacles_y"]) / 2

        if np.isnan([x1,y1,x2,y2,x3,y3,x4,y4]).any():
            nan_count += 1; continue

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if denom == 0:
            parallel_count += 1; continue

        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom

        dist_inter = np.sqrt((px-x1)**2 + (py-y1)**2)
        shrimp_len = np.sqrt((f["HeadShrimp_x"]-f["TailShrimp_x"])**2 +
                            (f["HeadShrimp_y"]-f["TailShrimp_y"])**2)

        if shrimp_len > 0 and not np.isnan(dist_inter):
            Data.loc[idx, "distance_intersection_normalized"] = dist_inter / shrimp_len
            ok_count += 1
        else:
            nan_count += 1

    print(f"\nHit point — OK:{ok_count} NoMatch:{no_match_count} "
        f"Parallèles:{parallel_count} NaN:{nan_count}")

    Data.to_excel(r"E:/01_Travail/CREC/First_Capture/analyse/Data_cuttlfish_strike_with_intersection.xlsx",
                index=False)
    print("[OK] Hit point sauvegardé")