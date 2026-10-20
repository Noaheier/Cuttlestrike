"""
add_features_full.py
Même logique que add_features_cut_2.py, au choix (SOURCE) :
  - "full" : tracking COMPLET (combined_cuttlefish_tracking.csv), chaque
             individu gardé depuis Start jusqu'à la frappe (Catch_frame_nb)
  - "cut"  : tracking CUT (cut_combined_cuttlefish_tracking.csv), déjà coupé
             de Frame_start à la frappe -> mêmes variables, sur le segment cut

Repère des frames : le tracking a été fait sur les vidéos 01_VIDEO_Cropped
(vidéo d'origine coupée de Start à End). Frame 0 = Start, et Catch_frame_nb
est exprimé dans ce repère -> chaque individu = frames 0 .. Catch_frame_nb
de sa vidéo Cropped (Catch_frame_nb + 1 frames).

Différences avec le fichier "cut" :
  - le modèle DLC du fichier complet n'a PAS MidCuttleHead ni les tentacules
    -> pas de calcul "hit point" ici
  - beaucoup plus de frames (~1.3 M jusqu'à la frappe contre ~95 k)
  - nouvelles variables de VARIATION sur une fenêtre (VAR_WINDOW)
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
SOURCE = "cut"    # "full" = combined_cuttlefish_tracking     (Start -> frappe, vidéos 01_VIDEO_Cropped)
                   # "cut"  = cut_combined_cuttlefish_tracking (Frame_start -> frappe, vidéos *_cut.mp4)

TRACKING_PATH = {
    "full": DATA_DIR / "combined_cuttlefish_tracking.csv",   # NON inclus (662 Mo) : à placer dans Data/
    "cut":  DATA_DIR / "cut_combined_cuttlefish_tracking.csv",
}[SOURCE]
SAVE_TEMPLATE = {
    "full": str(DATA_DIR / "full_cuttlefish_with_features_smooth_{w}.csv"),
    # nom différent pour ne PAS écraser cut_cuttlefish_with_features_smooth_*.csv (ancien calcul)
    "cut":  str(DATA_DIR / "cut_cuttlefish_with_new_features_smooth_{w}.csv"),
}[SOURCE]
EXCEL_PATH    = DATA_DIR / "Data_cuttlfish_strike.xlsx"

ALL_SMOOTH_WINDOW  = (1,10,30)      # ex. (1, 5, 10) — chaque fenêtre = un CSV (fichiers lourds)
START_AT           = "video"   # SOURCE "full" seulement : "video" = frame 0 = Start
                               #   (le tracking vient des vidéos 01_VIDEO_Cropped, déjà
                               #   coupées de Start à End) | "cut" = Frame_start
POST_STRIKE_FRAMES = 0         # SOURCE "full" seulement : frames gardées APRÈS la frappe
VAR_WINDOW         = 20        # fenêtre (frames) des variables de VARIATION ; 20 f = 1/3 s à 60 fps
SPEED_WINDOW       = 10        # lissage des vitesses (frames)
MAX_JUMP_PX        = 50        # déplacement max physiquement possible en 1 frame (1/60 s)
MAX_GLITCH_FRAMES  = 30        # au-delà, une position "trop loin" est acceptée (vrai déplacement)
SPEED_NOISE_MM_S   = 0.5       # en dessous : bruit de tracking -> vitesse mise à 0
ANGLE_NOISE_DEG    = 0.15      # variation d'angle < 0.15°/frame = bruit -> ignorée dans les sommes
SHRIMP_MIN_LEN     = 0.6       # axe crevette rejeté si tête-milieu < 60 % de sa longueur habituelle
ORBIT_MIN_SPEED    = 10.0      # orbit_speed : cap de la trajectoire défini seulement au-dessus (mm/s)
ORBIT_SMOOTH       = 30        # orbit_speed : lissage final (frames, 0.5 s) -> épisodes continus
ORBIT_MIN_DEG_S    = 0.0       # orbit_speed : en dessous (°/s) = légère courbe, pas un contournement -> 0
                               #   (0 = pas de seuil ; la valeur brute reste dans orbit_speed_deg_s)
TEST_N_ANIMALS     = None      # ex. 5 pour tester rapidement sur quelques individus

POINTS = ("Apex", "TopMantle", "HeadShrimp", "MidShrimp", "TailShrimp")   # points utilisés


# ══════════════════════════════════════════════════════════════
# FONCTIONS
# ══════════════════════════════════════════════════════════════
def valid_track(x, y, max_jump, max_glitch):
    """
    Marque les positions aberrantes en comparant chaque frame à la DERNIÈRE
    POSITION VALIDE (pas à la frame précédente) : si le point saute puis reste
    faux plusieurs frames, toutes ces frames sont rejetées. La distance
    autorisée grandit avec le nombre de frames écoulées (max_jump par frame) ;
    après max_glitch frames, la nouvelle position est acceptée.
    """
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
    """Positions x, y d'un point avec les sauts aberrants retirés puis interpolés.
    Renvoie aussi le nombre de positions rejetées."""
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


def speed_mm_s(vx, vy):
    """Norme de la vitesse en mm/s, écrêtée au p99.5, bruit (< SPEED_NOISE_MM_S) -> 0."""
    v = np.sqrt(vx ** 2 + vy ** 2) * MM * FPS
    v = v.clip(upper=v.quantile(0.995))
    v[v < SPEED_NOISE_MM_S] = 0.0
    return v


def centered_change(df, col, window):
    """x(t + w/2) - x(t - w/2), par individu. Aux bords : fenêtre tronquée."""
    h = max(window // 2, 1)
    g = df.groupby("animal_id")[col]
    fwd = g.shift(-h).fillna(g.transform("last"))
    bwd = g.shift(h).fillna(g.transform("first"))
    return fwd - bwd


def rolling_abs_change(df, series, window, circular=False, noise=0.0):
    """Somme des |variations| frame à frame sur une fenêtre centrée.
    Les variations dans les deux sens comptent (ajustements fins) ; celles
    inférieures à `noise` (même unité que series) sont du bruit -> 0."""
    d = series.groupby(df["animal_id"]).diff()
    if circular:                                  # angles en radians
        d = (d + np.pi) % (2 * np.pi) - np.pi
    d = d.abs().fillna(0)
    d[d < noise] = 0.0
    return d.groupby(df["animal_id"]).transform(
        lambda s: s.rolling(window, center=True, min_periods=1).sum())


def angle_between_vectors(dx1, dy1, dx2, dy2):
    """Angle SANS signe entre deux vecteurs (arccos, 0..180°)."""
    dot  = dx1 * dx2 + dy1 * dy2
    norm = np.sqrt(dx1 ** 2 + dy1 ** 2) * np.sqrt(dx2 ** 2 + dy2 ** 2)
    return np.degrees(np.arccos(np.clip(dot / (norm + 1e-9), -1.0, 1.0)))


# ── 1. Chargement + découpe début -> frappe ───────────────────
raw  = pd.read_csv(TRACKING_PATH, on_bad_lines="skip")
Data = pd.read_excel(EXCEL_PATH)
raw["animal_id"] = raw["animal_id"].astype(int)
print(f"[OK] Chargé ({SOURCE}) : {len(raw)} frames | {raw['animal_id'].nunique()} individus")

info = Data.set_index(Data["num"].astype(int))[
    ["Catch_frame_nb", "Frame_start", "scale_1px_mm", "Fps"]].copy()
missing = sorted(set(raw["animal_id"]) - set(info.index))
if TEST_N_ANIMALS:
    raw = raw[raw["animal_id"].isin(sorted(raw["animal_id"].unique())[:TEST_N_ANIMALS])]

if SOURCE == "full":
    # Repère des vidéos 01_VIDEO_Cropped (frame 0 = Start). Frame_start et
    # Catch_frame_nb sont dans ce repère -> découpe start -> frappe.
    if missing:
        print(f"[WARN] {len(missing)} individus absents de l'Excel (pas de frame de frappe) "
              f"-> ignorés : {missing}")
    raw = raw[raw["animal_id"].isin(info.index)]
    info["start_frame"] = info["Frame_start"] if START_AT == "cut" else 0
    strike = raw["animal_id"].map(info["Catch_frame_nb"])
    start  = raw["animal_id"].map(info["start_frame"])
    raw = raw[(raw["frame"] >= start) & (raw["frame"] <= strike + POST_STRIKE_FRAMES)]
    raw = raw.sort_values(["animal_id", "frame"]).reset_index(drop=True)
    raw["strike_frame"]   = raw["animal_id"].map(info["Catch_frame_nb"])
    raw["in_cut_segment"] = raw["frame"] >= raw["animal_id"].map(info["Frame_start"])
    print(f"[OK] Après découpe ({START_AT} -> frappe + {POST_STRIKE_FRAMES}) : "
          f"{len(raw)} frames | {raw['animal_id'].nunique()} individus")
else:
    # Fichier cut : déjà coupé de Frame_start à la frappe, frames renumérotées
    # 0..n (repère des vidéos *_cut.mp4) -> la frappe = dernière frame.
    # Tous les individus sont gardés (comme l'ancien fichier cut) ; ceux absents
    # de l'Excel prennent l'échelle et le fps médians.
    raw = raw.sort_values(["animal_id", "frame"]).reset_index(drop=True)
    raw["strike_frame"]   = raw.groupby("animal_id")["frame"].transform("max")
    raw["in_cut_segment"] = True
    if missing:
        print(f"[WARN] {len(missing)} individus absents de l'Excel -> échelle et fps médians : {missing}")
    n_cut    = raw.groupby("animal_id").size()
    expected = (info["Catch_frame_nb"] - info["Frame_start"] + 1).reindex(n_cut.index)
    odd = n_cut.index[expected.notna() & (n_cut != expected)].tolist()
    if odd:
        print(f"[WARN] longueur différente de Catch_frame_nb - Frame_start + 1 dans l'Excel "
              f"(vidéo cut à vérifier) : {odd}")
    info = info.reindex(sorted(set(info.index) | set(raw["animal_id"])))
    for c in ("scale_1px_mm", "Fps"):
        info[c] = info[c].fillna(info[c].median())

# repères temporels et échelle (px -> mm, frame -> s)
raw["frame_rel_strike"] = raw["frame"] - raw["strike_frame"]           # 0 = frappe, < 0 avant
raw["time_to_strike_s"] = raw["frame_rel_strike"] / raw["animal_id"].map(info["Fps"])
MM  = raw["animal_id"].map(info["scale_1px_mm"])                        # mm par pixel
FPS = raw["animal_id"].map(info["Fps"])


for SMOOTH_WINDOW in ALL_SMOOTH_WINDOW:
    print(f"\n=== SMOOTH_WINDOW={SMOOTH_WINDOW} frames | VAR_WINDOW={VAR_WINDOW} frames")
    SAVE_PATH = SAVE_TEMPLATE.format(w=SMOOTH_WINDOW)

    # ── 2. Lissage DLC avec protection des bords ──────────────
    df = raw.copy()
    w  = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW + 1
    if w > 1:
        for col in [c for c in df.columns if c.endswith(("_x", "_y"))]:
            df[col] = df.groupby("animal_id")[col].transform(lambda s: smooth_group(s, w))

    # ── 3. Retrait des sauts de tracking sur tous les points utilisés ─
    # (après le lissage) -> toutes les métriques suivantes utilisent ces positions
    rejected = {}
    for pt in POINTS:
        df[f"{pt}_x"], df[f"{pt}_y"], rejected[pt] = clean_positions(df, pt)
    print("[sauts] positions rejetées : " +
          " | ".join(f"{pt} {n} ({n / len(df):.2%})" for pt, n in rejected.items()))

    # axe du corps de la seiche (Apex -> TopMantle), utilisé par plusieurs métriques
    ax_x = df["TopMantle_x"] - df["Apex_x"]
    ax_y = df["TopMantle_y"] - df["Apex_y"]
    ax_n = np.sqrt(ax_x ** 2 + ax_y ** 2) + 1e-9

    # ── 4. Distance seiche–crevette (+ ses VARIATIONS) ────────
    df["distance_to_shrimp_raw"] = np.sqrt((df["TopMantle_x"] - df["MidShrimp_x"]) ** 2 +
                                           (df["TopMantle_y"] - df["MidShrimp_y"]) ** 2)
    df["distance_to_shrimp"] = np.log1p(df["distance_to_shrimp_raw"])     # comme le cut
    df["distance_mm"]        = df["distance_to_shrimp_raw"] * MM

    # Variations sur la fenêtre VAR_WINDOW, exprimées en mm/s (indépendant de la fenêtre)
    dt = 2 * max(VAR_WINDOW // 2, 1) / FPS                                  # durée de la fenêtre (s)
    #   distance_change : vitesse de variation de la distance
    #     < 0 = les animaux se rapprochent, > 0 = ils s'éloignent
    df["distance_change"] = centered_change(df, "distance_mm", VAR_WINDOW) / dt
    #   on sépare QUI fait varier la distance : déplacements projetés sur l'axe
    #   seiche -> crevette (au temps t)
    ux = df["MidShrimp_x"] - df["TopMantle_x"]
    uy = df["MidShrimp_y"] - df["TopMantle_y"]
    un = np.sqrt(ux ** 2 + uy ** 2) + 1e-9
    d_tx, d_ty = centered_change(df, "TopMantle_x", VAR_WINDOW), centered_change(df, "TopMantle_y", VAR_WINDOW)
    d_sx, d_sy = centered_change(df, "MidShrimp_x", VAR_WINDOW), centered_change(df, "MidShrimp_y", VAR_WINDOW)
    #   cuttle_approach_speed : la seiche avance VERS la crevette (> 0) ou recule (< 0)
    df["cuttle_approach_speed"] = (d_tx * ux + d_ty * uy) / un * MM / dt
    #   orbit_speed_old : ANCIENNE orbite (gardée pour comparaison, °/s)
    #     déplacement de TopMantle PERPENDICULAIRE à l'axe seiche -> crevette, divisé
    #     par la distance. Défaut : une ligne droite qui passe à côté de la proie
    #     compte comme une orbite (36 % des fortes valeurs). Nouvelle version : section 5.
    tang_mm_s = (d_tx * (-uy) + d_ty * ux) / un * MM / dt              # vitesse latérale (mm/s)
    orbit = np.degrees(np.abs(tang_mm_s) / (df["distance_mm"] + 1e-9))
    orbit[np.abs(tang_mm_s) < SPEED_NOISE_MM_S] = 0.0                  # bruit de tracking -> 0
    df["orbit_speed_old"] = orbit
    #   shrimp_retreat_speed  : la crevette s'ÉLOIGNE de la seiche (> 0) ou s'en rapproche (< 0)
    df["shrimp_retreat_speed"]  = (d_sx * ux + d_sy * uy) / un * MM / dt

    # ── 5. Vitesses (seiche = centre du corps, crevette = tête) ─
    # centre du corps : insensible aux rotations sur place
    df["body_cx"] = (df["Apex_x"] + df["TopMantle_x"]) / 2
    df["body_cy"] = (df["Apex_y"] + df["TopMantle_y"]) / 2
    vx, vy = velocity(df, "body_cx", "body_cy")
    df["speed_cuttle_mm_s"]   = speed_mm_s(vx, vy)                       # mm/s
    df["speed_cuttle_log"]    = np.log1p(df["speed_cuttle_mm_s"])        # pour le clustering
    df["speed_cuttle_smooth"] = df["speed_cuttle_log"]                   # alias (compatibilité notebook)

    # orbit_speed : la seiche TOURNE AUTOUR de la crevette (°/s, sans signe)
    #   = angle balayé autour de la crevette par le SEUL déplacement de la seiche
    #     (centre du corps), compté seulement si la TRAJECTOIRE tourne dans le même
    #     sens : min(angle balayé, virage du cap) sur VAR_WINDOW, puis lissage.
    #   -> 0 pour une ligne droite qui passe à côté de la proie, un pivot sur place
    #      ou un mouvement de la crevette seule.
    gid  = df["animal_id"]
    roll = lambda s, w, how: s.groupby(gid).transform(
        lambda x: getattr(x.rolling(w, center=True, min_periods=1), how)())
    bvx = roll(df["body_cx"].groupby(gid).diff().fillna(0), 5, "mean")
    bvy = roll(df["body_cy"].groupby(gid).diff().fillna(0), 5, "mean")
    rx, ry = df["body_cx"] - df["MidShrimp_x"], df["body_cy"] - df["MidShrimp_y"]
    d_theta = np.degrees((rx * bvy - ry * bvx) / (rx ** 2 + ry ** 2 + 1e-9))   # angle balayé (°/frame)
    heading = np.degrees(np.arctan2(bvy, bvx))
    moving  = df["speed_cuttle_mm_s"] > ORBIT_MIN_SPEED                     # cap fiable
    d_head  = (heading.groupby(gid).diff() + 180) % 360 - 180               # virage (°/frame)
    d_head  = d_head.where(moving & moving.groupby(gid).shift(1, fill_value=False), 0).fillna(0)
    D_TH, D_HE = roll(d_theta, VAR_WINDOW, "sum"), roll(d_head, VAR_WINDOW, "sum")
    orbit = pd.Series(np.where(np.sign(D_TH) == np.sign(D_HE),
                               np.minimum(D_TH.abs(), D_HE.abs()), 0.0), index=df.index)
    df["orbit_speed"] = roll(orbit / (VAR_WINDOW / FPS), ORBIT_SMOOTH, "mean")

    vx, vy = velocity(df, "HeadShrimp_x", "HeadShrimp_y")
    df["speed_shrimp_mm_s"]   = speed_mm_s(vx, vy)
    df["speed_shrimp_log"]    = np.log1p(df["speed_shrimp_mm_s"])
    df["speed_shrimp_smooth"] = df["speed_shrimp_log"]                   # alias (compatibilité)

    # who_moves : QUI bouge ? log(1 + vitesse seiche) - log(1 + vitesse crevette)
    #   très positif = la seiche bouge, la crevette peu (approche active)
    #   très négatif = la crevette bouge, la seiche reste immobile (affût)
    #   ≈ 0          = les deux immobiles, ou bougent autant (poursuite)
    df["who_moves"] = df["speed_cuttle_log"] - df["speed_shrimp_log"]

    # ── 6. Angles ──────────────────────────────────────────────
    # approach_angle : axe du corps vs axe de la crevette (comme le cut)
    # angle SANS signe (arccos, 0..180°) -> pas de saut artificiel 0 <-> 180
    sh_x = df["HeadShrimp_x"] - df["MidShrimp_x"]
    sh_y = df["HeadShrimp_y"] - df["MidShrimp_y"]
    df["approach_angle"] = angle_between_vectors(ax_x, ax_y, sh_x, sh_y)
    # Inversions tête <-> milieu de la crevette (DLC, crevette repliée) : la
    # distance tête-milieu s'effondre et l'angle bascule en 180° - angle.
    # -> angle rejeté puis interpolé quand tête-milieu < SHRIMP_MIN_LEN × habituelle
    sh_len  = np.sqrt(sh_x ** 2 + sh_y ** 2)
    flipped = sh_len < SHRIMP_MIN_LEN * sh_len.groupby(df["animal_id"]).transform("median")
    df.loc[flipped, "approach_angle"] = np.nan
    df["approach_angle"] = df.groupby("animal_id")["approach_angle"].transform(
        lambda s: s.interpolate("linear").ffill().bfill())
    print(f"[crevette] axe rejeté (tête-milieu trop court) : {flipped.sum()} frames ({flipped.mean():.2%})")

    # attack_side : OÙ est la seiche autour de la crevette ?
    #   angle entre l'axe de la crevette (milieu -> tête) et le vecteur crevette -> seiche
    #   0° = devant la tête de la crevette, 90° = sur le côté, 180° = derrière sa queue
    #   (même filtre des inversions tête <-> milieu que approach_angle)
    df["attack_side"] = angle_between_vectors(sh_x, sh_y, -ux, -uy)
    df.loc[flipped, "attack_side"] = np.nan
    df["attack_side"] = df.groupby("animal_id")["attack_side"].transform(
        lambda s: s.interpolate("linear").ffill().bfill())
    # approach_angle_variation : somme des |Δ| sur la fenêtre (deg), dans les deux
    # sens ; les |Δ| < ANGLE_NOISE_DEG par frame sont ignorés (bruit)
    df["approach_angle_variation"] = rolling_abs_change(df, df["approach_angle"], VAR_WINDOW,
                                                        noise=ANGLE_NOISE_DEG)

    # bearing : la seiche pointe-t-elle VERS la crevette ?
    #   angle entre l'axe du corps et le vecteur seiche -> crevette
    #   0° = la crevette est droit devant, 180° = derrière
    df["bearing_to_prey"] = angle_between_vectors(ax_x, ax_y, ux, uy)

    # orientation_variation : rotation totale du corps sur la fenêtre (deg)
    #   angle complet 360° (passage -180/+180 corrigé), rotations dans les deux
    #   sens comptées, |Δ| < ANGLE_NOISE_DEG par frame ignorés (bruit)
    body_angle = np.arctan2(ax_y, ax_x)
    df["orientation_variation"] = rolling_abs_change(df, body_angle, VAR_WINDOW, circular=True,
                                                     noise=np.radians(ANGLE_NOISE_DEG)) * 180 / np.pi
    # écrêtage au p99.5 seulement : au p95 les réorientations brusques d'avant
    # la frappe étaient toutes plafonnées à la même valeur
    df["orientation_variation"] = df["orientation_variation"].clip(
        upper=df["orientation_variation"].quantile(0.995))

    # ── 7. Écrêtage des valeurs extrêmes (sauts de tracking DLC) ─
    #   variables signées : percentiles 1 / 99 ; variations ≥ 0 : percentile 99
    for f in ("distance_change", "cuttle_approach_speed", "shrimp_retreat_speed"):
        df[f] = df[f].clip(df[f].quantile(0.01), df[f].quantile(0.99))
    for f in ("approach_angle_variation", "orbit_speed", "orbit_speed_old"):
        df[f] = df[f].clip(upper=df[f].quantile(0.99))

    # ── 7b. Versions LOG pour le clustering (distributions très asymétriques :
    #   beaucoup de 0 et quelques valeurs extrêmes). Les valeurs brutes sont
    #   gardées avec leur unité dans le nom.
    df["orbit_speed_deg_s"]          = df["orbit_speed"]                         # °/s
    #   seuil ORBIT_MIN_DEG_S avant le log : sans lui, log1p rend une légère courbe
    #   (3-6 °/s) presque aussi forte qu'un vrai contournement (15-50 °/s)
    df["orbit_speed"]                = np.log1p(np.clip(df["orbit_speed_deg_s"] - ORBIT_MIN_DEG_S, 0, None))
    df["cuttle_approach_speed_mm_s"] = df["cuttle_approach_speed"]               # mm/s, signée
    #   log SIGNÉ : garde le sens (+ avance / - recule), compresse l'amplitude
    df["cuttle_approach_speed"]      = (np.sign(df["cuttle_approach_speed_mm_s"])
                                        * np.log1p(df["cuttle_approach_speed_mm_s"].abs()))

    # ── 8. Vérification ────────────────────────────────────────
    features = ["speed_cuttle_log", "speed_shrimp_log", "distance_to_shrimp",
                "approach_angle", "approach_angle_variation", "orientation_variation",
                "distance_change", "cuttle_approach_speed", "shrimp_retreat_speed",
                "bearing_to_prey", "attack_side", "orbit_speed", "orbit_speed_old", "who_moves"]
    print("\n=== NaN / Inf / résumé ===")
    for f in features:
        n_nan, n_inf = df[f].isna().sum(), np.isinf(df[f]).sum()
        ok = "✅" if n_nan == 0 and n_inf == 0 else "❌"
        print(f"  {ok} {f:<26} NaN={n_nan:<6} Inf={n_inf:<4} "
              f"min={df[f].min():9.3f}  médiane={df[f].median():9.3f}  max={df[f].max():9.3f}")

    corr = df[features].corr().round(2)
    print("\nPaires très corrélées (|r| > 0.7) — redondance possible :")
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            if abs(corr.loc[f1, f2]) > 0.7:
                print(f"  ⚠️  {f1} ↔ {f2} : r={corr.loc[f1, f2]:.2f}")

    # ── 9. Sauvegarde ─────────────────────────────────────────
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n[OK] CSV sauvegardé → {SAVE_PATH}  ({len(df)} frames, {df.shape[1]} colonnes)")
