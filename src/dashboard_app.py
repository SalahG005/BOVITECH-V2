"""
Bovitech dashboard: metrics, prediction CSV explorer, accelerometer simulator.

Run from project root:
  streamlit run src/dashboard_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Project root = parent of src/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from predict_core import load_model_bundle, predict_from_immu  # noqa: E402

# Libellés par défaut (1–7) si behavior_map.json absent — à aligner avec ton projet
_DEFAULT_BEHAVIOR_LABELS: dict[int, str] = {
    1: "Walking",
    2: "Standing",
    3: "Feeding head up",
    4: "Feeding head down",
    5: "Licking",
    6: "Drinking",
    7: "Lying",
}


def _merged_behavior_labels(bmap: dict | None) -> dict[int, str]:
    out = dict(_DEFAULT_BEHAVIOR_LABELS)
    if bmap:
        out.update(bmap)
    return out


def _enrich_pred_columns(df: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    """Ajoute *_name pour pred_behavior et pred_behavior_smooth."""
    out = df.copy()
    for col in ("pred_behavior_smooth", "pred_behavior"):
        if col not in out.columns:
            continue
        name_col = col + "_name"
        def _lbl(v: object) -> str:
            if pd.isna(v):
                return ""
            try:
                k = int(round(float(v)))
                return labels.get(k, str(k))
            except (TypeError, ValueError):
                return str(v)

        out[name_col] = out[col].apply(_lbl)
    return out


def _percent_time_by_class(series: pd.Series, labels: dict[int, str]) -> pd.DataFrame:
    """Pourcentage du nombre de lignes (secondes) par classe."""
    s = pd.to_numeric(series, errors="coerce").dropna().astype("int64")
    if len(s) == 0:
        return pd.DataFrame(columns=["Comportement", "% du temps"])
    vc = s.value_counts(normalize=True).sort_index() * 100.0
    rows = [{"Comportement": labels.get(int(k), str(k)), "% du temps": round(float(v), 2)} for k, v in vc.items()]
    return pd.DataFrame(rows)


def _render_behavior_breakdown(df: pd.DataFrame, labels: dict[int, str], title: str) -> None:
    st.subheader(title)
    col_pred = None
    for c in ("pred_behavior_smooth", "pred_behavior"):
        if c in df.columns:
            col_pred = c
            break
    if col_pred is None:
        st.caption("Aucune colonne pred_behavior trouvée.")
        return
    pct_df = _percent_time_by_class(df[col_pred], labels)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(pct_df, use_container_width=True, hide_index=True)
    with c2:
        try:
            import plotly.express as px

            if len(pct_df) > 0:
                ymax = max(100.0, float(pct_df["% du temps"].max()) * 1.15)
                fig = px.bar(
                    pct_df,
                    x="Comportement",
                    y="% du temps",
                    title="% du temps par comportement (fichier entier)",
                    text="% du temps",
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(xaxis_tickangle=-35, yaxis_range=[0, ymax])
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            if len(pct_df) > 0:
                st.bar_chart(pct_df.set_index("Comportement")["% du temps"])

    if "ts_sec" not in df.columns:
        return
    dt = pd.to_datetime(df["ts_sec"], unit="s", utc=False)
    by_day = df.assign(_date=dt.dt.date)
    dates = sorted(by_day["_date"].dropna().unique())
    if len(dates) == 0:
        return
    st.markdown("**Par jour (date calendaire dérivée de `ts_sec`)** — ex. une journée type `0721` dans le fichier")
    for d in dates:
        sub = by_day[by_day["_date"] == d]
        p2 = _percent_time_by_class(sub[col_pred], labels)
        with st.expander(
            f"{d} — {len(sub)} s — % par comportement",
            expanded=(len(dates) == 1),
        ):
            st.dataframe(p2, use_container_width=True, hide_index=True)
            try:
                import plotly.express as px

                if len(p2) > 0:
                    fig2 = px.bar(
                        p2,
                        x="Comportement",
                        y="% du temps",
                        title=f"{d} — % du temps",
                        text="% du temps",
                    )
                    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig2.update_layout(xaxis_tickangle=-35)
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass


st.set_page_config(page_title="Bovitech — Comportement vache", layout="wide")


def _default_model_dir() -> Path:
    p = Path(r"C:\bovitech_artifacts\model")
    if p.exists():
        return p
    return _ROOT / "artifacts" / "model"


with st.sidebar:
    st.header("Configuration")
    model_dir = st.text_input("Dossier modèle (`model_dir`)", value=str(_default_model_dir()))
    sensor_root = st.text_input(
        "Racine capteurs (optionnel)",
        value=r"C:\sensor_data\sensor_data",
        help="Pour charger behavior_map.json par défaut",
    )
    behavior_map_path = st.text_input(
        "behavior_map.json (optionnel)",
        value="",
        help="Laisse vide pour essayer artifacts/model/behavior_map.json",
    )
    smooth_default = st.slider("Lissage prédictions (secondes)", 1, 15, 7)


def load_behavior_map_optional() -> dict | None:
    if behavior_map_path.strip():
        p = Path(behavior_map_path.strip())
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in raw.items()}
    for candidate in (
        Path(model_dir) / "behavior_map.json",
        _ROOT / "artifacts" / "model" / "behavior_map.json",
    ):
        if candidate.exists():
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in raw.items()}
    return None


tab_metrics, tab_pred, tab_sim = st.tabs(
    ["Résultats & métriques", "Fichier prédictions", "Simulateur accéléromètre (x,y,z)"]
)

with tab_metrics:
    st.subheader("Modèle et métriques d’entraînement")
    mp = Path(model_dir)
    if not mp.exists():
        st.error(f"Dossier introuvable: {mp}")
    else:
        meta_files = list(mp.glob("metadata*.json"))
        if not meta_files:
            st.warning("Aucun metadata*.json trouvé.")
        else:
            choice = st.selectbox("Fichier metadata", [str(p) for p in sorted(meta_files)])
            meta = json.loads(Path(choice).read_text(encoding="utf-8"))
            st.json({k: meta[k] for k in ("metrics", "history_seconds", "include_mag", "cows", "dates", "model_params") if k in meta})
            if "feature_columns" in meta:
                st.caption(f"Nombre de features: {len(meta['feature_columns'])}")

        fi = mp / "feature_importance_multimodal.csv"
        if not fi.exists():
            fi = mp / "feature_importance_immu.csv"
        if fi.exists():
            st.subheader("Importance des variables (top 25)")
            fdf = pd.read_csv(fi).head(25)
            st.dataframe(fdf, use_container_width=True)
            try:
                import plotly.express as px

                fig = px.bar(fdf, x="importance", y="feature", orientation="h", title="Feature importance")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        cm_candidates = [mp / "confusion_matrix_multimodal.csv", mp / "confusion_matrix_immu.csv"]
        cm_path = next((p for p in cm_candidates if p.exists()), None)
        if cm_path:
            st.subheader("Matrice de confusion (CSV sauvegardé à l’entraînement)")
            cm = pd.read_csv(cm_path)
            st.dataframe(cm, use_container_width=True)

with tab_pred:
    st.subheader("Visualiser un CSV de prédictions")
    uploaded = st.file_uploader("Uploader un CSV (ts_sec, pred_behavior, …)", type=["csv"])
    pred_path = st.text_input("Ou chemin vers un CSV existant", value=str(_ROOT / "artifacts" / "predictions" / "T09_0725_pred_final.csv"))
    df_plot = None
    if uploaded is not None:
        df_plot = pd.read_csv(uploaded)
    elif Path(pred_path).exists():
        df_plot = pd.read_csv(pred_path)
    if df_plot is not None and len(df_plot) > 0:
        labels = _merged_behavior_labels(load_behavior_map_optional())
        df_view = _enrich_pred_columns(df_plot, labels)
        show_cols = [c for c in df_view.columns if c in ("ts_sec", "pred_behavior_smooth_name", "pred_behavior_name", "pred_behavior_smooth", "pred_behavior")]
        other = [c for c in df_view.columns if c not in show_cols]
        st.dataframe(df_view[show_cols + other].head(200), use_container_width=True)
        col_ts = "ts_sec" if "ts_sec" in df_plot.columns else df_plot.columns[0]
        col_pred = None
        for c in ("pred_behavior_smooth", "pred_behavior"):
            if c in df_plot.columns:
                col_pred = c
                break
        if col_pred:
            try:
                import plotly.graph_objects as go

                yv = pd.to_numeric(df_plot[col_pred], errors="coerce").dropna().astype(int)
                u = sorted(yv.unique())
                tick_vals = u
                tick_text = [labels.get(k, str(k)) for k in u]

                if col_pred + "_name" in df_view.columns:
                    hover_arr = df_view[col_pred + "_name"].tolist()
                else:
                    hover_arr = df_plot[col_pred].apply(
                        lambda x: labels.get(int(round(float(x))), str(x)) if pd.notna(x) else ""
                    ).tolist()

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_plot[col_ts],
                        y=df_plot[col_pred],
                        mode="lines",
                        name=col_pred,
                        line=dict(width=1),
                        hovertext=hover_arr,
                        hovertemplate="%{hovertext}<br>"
                        + col_ts
                        + "=%{x}<br>classe=%{y}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title="Comportement prédit dans le temps (axe Y = libellés)",
                    xaxis_title=col_ts,
                    yaxis_title="Comportement",
                    yaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Graphique indisponible: {e}")
        _render_behavior_breakdown(df_plot, labels, "Répartition du temps par comportement (% des secondes)")
    else:
        st.info("Charge un CSV ou indique un chemin valide.")

with tab_sim:
    st.markdown(
        """
Saisissez des échantillons **bruts** comme dans le fichier IMMU : au minimum  
`timestamp`, `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2` (plusieurs lignes par seconde possibles).  
Ajoutez `mag_x_uT`, `mag_y_uT`, `mag_z_uT` pour un **yaw** réaliste en mode synthèse tête.  
Le pipeline agrège **par seconde** puis applique le modèle (lags + multimodal si entraîné ainsi).

**Astuce :** si le modèle utilise des lags (`history_seconds` non nul dans les métadonnées), ajoute **plusieurs secondes consécutives** (ex. `1700000000`, `1700000001`, `1700000002`…) avec des IMU réalistes ; une seule seconde reste plus difficile à classer.
"""
    )
    try:
        _, _meta_sim = load_model_bundle(Path(model_dir))
        default_mm = _meta_sim.get("notes") == "multimodal"
        meta_head_src = _meta_sim.get("head_source")
    except Exception:
        default_mm = True
        meta_head_src = None
    st.caption(
        f"Modèle détecté comme multimodal (IMMU+Head) : **{default_mm}**"
        + (f" — entraînement tête : `{meta_head_src}`" if meta_head_src else "")
    )
    head_path_txt = st.text_input(
        "Optionnel — CSV direction de tête (si rempli et fichier existant : utilisé à la place de la synthèse IMU)",
        value="",
    )
    synth_head = st.checkbox(
        "Synthétiser la tête depuis l’IMU (défaut : oui si pas de CSV ci-dessus)",
        value=not head_path_txt.strip(),
        help="Décochez seulement si vous fournissez un CSV tête valide. Sinon roll/pitch/yaw viennent de imu_head_synthesis.",
    )

    st.subheader("Données IMMU")
    st.caption(
        "Éditez le tableau puis cliquez sur **Prédire le comportement** (bouton bleu sous la table)."
    )
    default_rows = pd.DataFrame(
        {
            "timestamp": [1700000000.0, 1700000000.02, 1700000000.04],
            "accel_x_mps2": [0.2, 0.3, 0.1],
            "accel_y_mps2": [-0.1, 0.0, 0.2],
            "accel_z_mps2": [9.7, 9.8, 9.6],
            "mag_x_uT": [22.0, 22.1, 21.9],
            "mag_y_uT": [-8.0, -8.1, -7.9],
            "mag_z_uT": [-42.0, -41.8, -42.2],
        }
    )

    with st.form("simulator_predict_form"):
        edited = st.data_editor(
            default_rows,
            num_rows="dynamic",
            use_container_width=True,
            key="sim_immu_editor",
        )
        submitted = st.form_submit_button(
            "Prédire le comportement",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if edited is None or len(edited) < 1:
            st.error("Ajoutez au moins une ligne.")
        else:
            missing = [c for c in ("timestamp", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2") if c not in edited.columns]
            if missing:
                st.error(f"Colonnes manquantes: {missing}")
            else:
                try:
                    bmap = load_behavior_map_optional()
                    head_p = Path(head_path_txt.strip()) if head_path_txt.strip() else None
                    if head_p is not None and not head_p.exists():
                        st.warning("Fichier tête introuvable — synthèse depuis l’IMMU.")
                        head_p = None
                    use_csv = head_p is not None and head_p.exists() and not synth_head
                    out = predict_from_immu(
                        model_dir=Path(model_dir),
                        immu_file=None,
                        immu_df=edited,
                        use_multimodal=default_mm,
                        head_file=head_p if use_csv else None,
                        smooth_window=int(smooth_default),
                        behavior_map=bmap,
                    )
                    st.success(f"{len(out)} seconde(s) prédite(s).")
                    lbl = _merged_behavior_labels(bmap)
                    out_named = _enrich_pred_columns(out, lbl)
                    show_c = [c for c in ("ts_sec", "pred_behavior_smooth_name", "pred_behavior_name", "pred_behavior_smooth", "pred_behavior") if c in out_named.columns]
                    rest = [c for c in out_named.columns if c not in show_c]
                    st.dataframe(out_named[show_c + rest], use_container_width=True)
                    _render_behavior_breakdown(out, lbl, "Répartition du temps (simulateur)")
                except Exception as e:
                    st.exception(e)
