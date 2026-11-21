"""Streamlit dashboard for the Federated Medical AI project.

Run from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.data.dataset import dataset_summary, load_dataset  # noqa: E402
from src.data.pipeline import partition_summary, prepare_data  # noqa: E402
from src.evaluation.visualization import PALETTE  # noqa: E402
from src.utils.config import load_config, override  # noqa: E402

st.set_page_config(page_title="Federated Medical AI", page_icon="🏥", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1300px;}
      div[data-testid="stMetric"] {background: rgba(42,120,214,0.06); border: 1px solid rgba(42,120,214,0.18);
           border-radius: 10px; padding: 12px 16px;}
      .note {border-left: 4px solid #eda100; padding: 0.6rem 1rem; background: rgba(237,161,0,0.08);
             border-radius: 6px; margin: 0.5rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

CFG = load_config()
METRICS_DIR = ROOT / CFG["paths"]["metrics_dir"]
FIG_DIR = ROOT / CFG["paths"]["figures_dir"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "roc_auc": "ROC-AUC"}


# ------------------------------------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def load_json(name: str) -> Any | None:
    p = METRICS_DIR / name
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def partitions(seed: int) -> dict[str, list[dict]]:
    return {"IID": partition_summary(prepare_data(CFG, "iid", seed)),
            "Non-IID": partition_summary(prepare_data(CFG, "non_iid", seed))}


def color(name: str) -> str:
    return PALETTE.get(name, "#52514e")


def style_fig(fig: go.Figure, height: int = 360, **layout: Any) -> go.Figure:
    base: dict[str, Any] = dict(height=height, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                                legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), hovermode="x unified")
    fig.update_layout(**{**base, **layout})
    return fig


def metric_tiles(m: dict[str, float]) -> None:
    cols = st.columns(5)
    for c, k in zip(cols, METRICS):
        c.metric(LABELS[k], f"{m[k]:.3f}")


def confusion_fig(cm: list[list[int]], title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred. benign", "Pred. malignant"], y=["True benign", "True malignant"],
                               colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}",
                               textfont={"size": 18}, hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 280, title=title, hovermode="closest")


def missing_results() -> None:
    st.warning("No saved results found. Run `python -m experiments.run_all` from the project root first.")


# ------------------------------------------------------------------------------------------------ pages
def page_overview() -> None:
    st.title("🏥 Federated Learning for Privacy-Preserving Medical AI")
    st.markdown(
        "Three simulated hospitals collaboratively train a breast-cancer diagnosis model **without sharing "
        "their patient records**. Each hospital trains a local PyTorch model on its private data and only "
        "sends model weights to a central server, which combines them with **Federated Averaging (FedAvg)** "
        "using Flower's `FedAvg` strategy. The global model is compared with centralized training (all data "
        "pooled) and with hospitals training alone."
    )
    full = dataset_summary(load_dataset())
    c = st.columns(4)
    c[0].metric("Hospitals", CFG["partition"]["num_clients"])
    c[1].metric("Patients (samples)", full["n_samples"])
    c[2].metric("Features", CFG["model"]["input_dim"])
    c[3].metric("Malignant ratio", f"{full['malignant_ratio']:.1%}")

    st.subheader("Architecture")
    names = CFG["partition"]["client_names"]
    hospitals = "\n".join(
        f'h{i} [label="{n}\\nprivate data + local model", shape=box, style="rounded,filled", fillcolor="#e8f1fb"];'
        f'h{i} -> s [label="  weights W_k, n_k", color="#1baf7a"]; s -> h{i} [label="  global W", color="#2a78d6"];'
        for i, n in enumerate(names)
    )
    st.graphviz_chart(
        'digraph { rankdir=LR; size="9,3.2"; node [fontname="Helvetica", fontsize=11]; edge [fontname="Helvetica", fontsize=9];'
        's [label="Central server\\nFlower FedAvg\\nW = Σ n_k W_k / Σ n_k", shape=box, style="rounded,filled", '
        f'fillcolor="#fdf0e0"]; {hospitals} }}',
        width="content",
    )

    st.subheader("Dataset")
    st.markdown(
        f"""
- **Breast Cancer Wisconsin (Diagnostic)** from `sklearn.datasets.load_breast_cancer`: 569 fine-needle-aspirate
  samples, 30 numeric features, binary label (**malignant = 1**, benign = 0).
- A stratified **{CFG['data']['test_size']:.0%} held-out test set** (114 samples) is never used for training,
  validation or scaling.
- The remaining 455 samples are split across hospitals, IID or **non-IID** (Dirichlet label skew,
  α = {CFG['partition']['dirichlet_alpha']}); each hospital keeps {CFG['data']['val_fraction']:.0%} for local validation.
"""
    )
    parts = partitions(CFG["seed"])
    cols = st.columns(2)
    for col, (title, rows) in zip(cols, parts.items()):
        df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_bar(x=df["hospital"], y=df["n_benign"], name="Benign", marker_color="#9ec5ef")
        fig.add_bar(x=df["hospital"], y=df["n_malignant"], name="Malignant", marker_color="#1c5cab")
        col.plotly_chart(style_fig(fig, 300, barmode="stack", title=f"{title} partition", hovermode="closest"),
                         width="stretch")


def page_training() -> None:
    st.title("⚙️ Run a federated experiment")
    st.caption("Runs a real Flower FedAvg simulation on CPU in this process (a few seconds).")
    with st.form("train"):
        c1, c2, c3, c4 = st.columns(4)
        mode = c1.radio("Data distribution", ["Non-IID", "IID"], horizontal=True)
        rounds = c2.slider("Communication rounds", 1, 20, CFG["federated"]["num_rounds"])
        local_epochs = c3.slider("Local epochs per round", 1, 10, CFG["federated"]["local_epochs"])
        seed = c4.number_input("Seed", 0, 10_000, CFG["seed"])
        submitted = st.form_submit_button("▶ Start experiment", type="primary")

    if submitted:
        from src.federated.simulation import run_federated

        cfg = override(CFG, **{"federated.local_epochs": int(local_epochs)})
        bar = st.progress(0.0, text="Initializing global model…")
        live = st.empty()
        rows: list[dict] = []

        def on_round(rnd: int, total: int, rec: dict) -> None:
            rows.append({"round": rnd, "train_loss": rec["train_loss"], "val_loss": rec["val_loss"],
                         "test_accuracy": rec.get("test_accuracy"), "test_roc_auc": rec.get("test_roc_auc")})
            bar.progress(rnd / total, text=f"Round {rnd}/{total} - train loss {rec['train_loss']:.4f}, "
                                           f"test accuracy {rec.get('test_accuracy', float('nan')):.3f}")
            live.dataframe(pd.DataFrame(rows).set_index("round"), width="stretch")

        res = run_federated(cfg, "iid" if mode == "IID" else "non_iid", num_rounds=int(rounds), seed=int(seed),
                            progress_callback=on_round)
        st.session_state["live_result"] = res
        bar.progress(1.0, text="Finished ✔")

    res = st.session_state.get("live_result")
    if res:
        st.subheader(f"Result: {res['name']}, {res['num_rounds']} rounds (seed {res['seed']})")
        metric_tiles(res["metrics"])
        hist = pd.DataFrame(res["history"])
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=hist["round"], y=hist["train_loss"], name="Train loss", mode="lines+markers",
                        line_color=color(res["name"]))
        fig.add_scatter(x=hist["round"], y=hist["val_loss"], name="Validation loss", mode="lines+markers",
                        line=dict(color=color(res["name"]), dash="dash"))
        c1.plotly_chart(style_fig(fig, title="Loss per round", xaxis_title="Round"), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=hist["round"], y=hist["test_accuracy"], name="Test accuracy", mode="lines+markers",
                        line_color=color(res["name"]))
        c2.plotly_chart(style_fig(fig, title="Global model test accuracy", xaxis_title="Round"),
                        width="stretch")
        c1, c2 = st.columns([1, 2])
        c1.plotly_chart(confusion_fig(res["metrics"]["confusion_matrix"], "Confusion matrix"), width="stretch")
        c2.dataframe(pd.DataFrame(res["partition"]), width="stretch", hide_index=True)
        comm = res["communication"]
        c2.info(f"{comm['num_messages']} messages, {comm['parameters_transmitted_total']:,} parameters, "
                f"{comm['kilobytes_total']:.1f} KB exchanged. Raw records transmitted: "
                f"{comm['raw_data_bytes_transmitted']} bytes. Payload audits passed: {comm['audits_passed']}.")


def page_results() -> None:
    st.title("📊 Results")
    main = load_json("main_results.json")
    if not main:
        return missing_results()
    st.caption(f"Saved results from `python -m experiments.run_all` (seed {main['seed']}); all metrics on the "
               "same 114-sample held-out test set.")
    runs = [main["centralized"], *main["local_only"], main["federated_iid"], main["federated_non_iid"]]
    choice = st.selectbox("Model", [r["name"] for r in runs], index=len(runs) - 1)
    run = next(r for r in runs if r["name"] == choice)
    metric_tiles(run["metrics"])

    table = pd.DataFrame([{"Model": r["name"], **{LABELS[k]: r["metrics"][k] for k in METRICS},
                           "Test loss": r["metrics"]["test_loss"]} for r in runs])
    st.dataframe(table.style.format(precision=4).highlight_max(subset=list(LABELS.values()), color="#d6ecdf"),
                 width="stretch", hide_index=True)

    rob = load_json("robustness.json")
    if rob:
        with st.expander(f"Robustness: mean ± std over seeds {rob['seeds']}", expanded=True):
            st.dataframe(pd.DataFrame([{"Model": n, **{LABELS[k]: f"{v[k]['mean']:.4f} ± {v[k]['std']:.4f}"
                                                       for k in METRICS}} for n, v in rob["summary"].items()]),
                         width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    fig = go.Figure()
    if "history" in run and isinstance(run["history"], dict):
        x = list(range(1, len(run["history"]["train_loss"]) + 1))
        fig.add_scatter(x=x, y=run["history"]["train_loss"], name="Train", line_color=color(run["name"]))
        fig.add_scatter(x=x, y=run["history"]["val_loss"], name="Validation",
                        line=dict(color=color(run["name"]), dash="dash"))
        xt = "Epoch"
    else:
        x = [h["round"] for h in run["history"]]
        fig.add_scatter(x=x, y=[h["train_loss"] for h in run["history"]], name="Train", mode="lines+markers",
                        line_color=color(run["name"]))
        fig.add_scatter(x=x, y=[h["val_loss"] for h in run["history"]], name="Validation", mode="lines+markers",
                        line=dict(color=color(run["name"]), dash="dash"))
        xt = "Communication round"
    c1.plotly_chart(style_fig(fig, title=f"Training curve - {run['name']}", xaxis_title=xt, yaxis_title="BCE loss"),
                    width="stretch")
    fig = go.Figure()
    for r in runs:
        roc = r["metrics"]["roc_curve"]
        fig.add_scatter(x=roc["fpr"], y=roc["tpr"], name=f"{r['name']} ({r['metrics']['roc_auc']:.3f})",
                        line=dict(color=color(r["name"]), width=2), visible=True if r is run or "Federated" in r["name"]
                        or r["name"] == "Centralized" else "legendonly")
    c2.plotly_chart(style_fig(fig, title="ROC curves (click legend to toggle)", xaxis_title="FPR",
                              yaxis_title="TPR", hovermode="closest"), width="stretch")

    st.subheader("Confusion matrices")
    cols = st.columns(3)
    for i, r in enumerate(runs):
        cols[i % 3].plotly_chart(confusion_fig(r["metrics"]["confusion_matrix"], r["name"]), width="stretch")

    st.subheader("Experiment E - number of communication rounds")
    rr = load_json("rounds_comparison.json") or []
    if rr:
        df = pd.DataFrame([{"Setting": r["name"], "Rounds": r["num_rounds"],
                            **{LABELS[k]: r["metrics"][k] for k in METRICS}, "Test loss": r["metrics"]["test_loss"],
                            "KB exchanged": r["communication"]["kilobytes_total"]} for r in rr])
        st.dataframe(df.style.format(precision=4), width="stretch", hide_index=True)
        longest = max(r["num_rounds"] for r in rr)
        fig = go.Figure()
        for r in [r for r in rr if r["num_rounds"] == longest]:
            fig.add_scatter(x=[h["round"] for h in r["history"]], y=[h["test_accuracy"] for h in r["history"]],
                            name=r["name"], mode="lines+markers", line_color=color(r["name"]))
        st.plotly_chart(style_fig(fig, title=f"Global test accuracy per round ({longest}-round runs)",
                                  xaxis_title="Round", yaxis_title="Accuracy"), width="stretch")

    with st.expander("Static figures (results/figures)"):
        for p in sorted(FIG_DIR.glob("*.png")):
            st.image(str(p), caption=p.name, width="stretch")


def page_hospitals() -> None:
    st.title("🏨 Hospital comparison")
    main = load_json("main_results.json")
    if not main:
        return missing_results()
    st.markdown("Each hospital's **local-only** model (trained only on its own non-IID data) versus the "
                "**federated global model** trained on the same non-IID partition. All evaluated on the shared "
                "held-out test set.")
    glob = main["federated_non_iid"]
    part = {p["hospital"]: p for p in glob["partition"]}
    rows = []
    for r in main["local_only"]:
        p = part[r["hospital"]]
        rows.append({"Model": r["name"], "Local train samples": r["n_train"],
                     "Malignant share": f"{p['malignant_ratio']:.0%}",
                     **{LABELS[k]: r["metrics"][k] for k in METRICS}})
    rows.append({"Model": glob["name"] + " (global)", "Local train samples": sum(p["n_train"] for p in glob["partition"]),
                 "Malignant share": "all hospitals", **{LABELS[k]: glob["metrics"][k] for k in METRICS}})
    st.dataframe(pd.DataFrame(rows).style.format(precision=4, subset=list(LABELS.values())),
                 width="stretch", hide_index=True)

    fig = go.Figure()
    for r in [*main["local_only"], glob]:
        fig.add_bar(x=[LABELS[k] for k in METRICS], y=[r["metrics"][k] for k in METRICS], name=r["name"],
                    marker_color=color(r["name"]))
    st.plotly_chart(style_fig(fig, 380, barmode="group", title="Local models vs. global federated model",
                              yaxis=dict(range=[0.75, 1.0]), hovermode="closest"), width="stretch")

    rob = load_json("robustness.json")
    if rob:
        names = [r["name"] for r in main["local_only"]] + ["Federated Non-IID"]
        fig = go.Figure()
        for n in names:
            s = rob["summary"][n]
            fig.add_bar(x=[LABELS[k] for k in METRICS], y=[s[k]["mean"] for k in METRICS], name=n,
                        marker_color=color(n), error_y=dict(type="data", array=[s[k]["std"] for k in METRICS]))
        st.plotly_chart(style_fig(fig, 380, barmode="group", title=f"Mean ± std over {len(rob['seeds'])} seeds",
                                  yaxis=dict(range=[0.75, 1.02]), hovermode="closest"), width="stretch")

    st.subheader("Per-hospital loss during federated training")
    fig = go.Figure()
    hcolors = {"Hospital A": color("Hospital A (local)"), "Hospital B": color("Hospital B (local)"),
               "Hospital C": color("Hospital C (local)")}
    for h in part:
        fig.add_scatter(x=[r["round"] for r in glob["history"]], y=[r["client_val_loss"][h] for r in glob["history"]],
                        name=f"{h} - validation loss of global model", mode="lines+markers",
                        line_color=hcolors.get(h, "#52514e"))
    st.plotly_chart(style_fig(fig, title="Global model evaluated on each hospital's private validation set",
                              xaxis_title="Round", yaxis_title="BCE loss"), width="stretch")


def page_privacy() -> None:
    st.title("🔒 Privacy analysis")
    st.markdown(
        """
**Data decentralization.** Patient records stay inside each hospital. In every round the server sends the
global model weights to each hospital and receives back only (i) updated weights, (ii) the number of local
training samples `n_k` and (iii) a scalar training loss. Feature standardization is also federated: each
hospital shares only its sample count, per-feature sums and sums of squares.
"""
    )
    comm = load_json("communication_summary.json")
    if not comm:
        return missing_results()
    which = st.radio("Run", ["federated_non_iid", "federated_iid"], horizontal=True,
                     format_func=lambda s: "Non-IID" if s.endswith("non_iid") else "IID")
    c = comm[which]
    cols = st.columns(4)
    cols[0].metric("Clients", c["num_clients"])
    cols[1].metric("Rounds", c["num_rounds"])
    cols[2].metric("Parameters per model", f"{c['parameters_per_model']:,}")
    cols[3].metric("Messages", c["num_messages"])
    cols = st.columns(4)
    cols[0].metric("Parameters transmitted", f"{c['parameters_transmitted_total']:,}")
    cols[1].metric("Total traffic", f"{c['kilobytes_total']:.1f} KB")
    cols[2].metric("Raw data transmitted", f"{c['raw_data_bytes_transmitted']} B")
    cols[3].metric("Payload audits passed", f"{c['num_audits']}/{c['num_audits']}" if c["audits_passed"] else "FAILED")
    st.markdown(
        f"Fields sent by hospitals: `{', '.join(c['fields_sent_by_clients'])}`. The raw training data kept locally "
        f"totals **{c['raw_data_bytes_total'] / 1024:.1f} KB** across hospitals, which is *less* than the "
        f"{c['kilobytes_total']:.1f} KB of model traffic: on such a tiny dataset FL is not a bandwidth saving. "
        "Its benefit here is that records never leave the hospital."
    )
    log_path = ROOT / CFG["paths"]["logs_dir"] / f"communication_log_{which.split('federated_')[1]}.json"
    if log_path.is_file():
        log = json.loads(log_path.read_text(encoding="utf-8"))
        msgs = pd.DataFrame(log["messages"])
        fig = go.Figure()
        for d, col in (("server->client", "#2a78d6"), ("client->server", "#1baf7a")):
            g = msgs[msgs["direction"] == d].groupby("round")["payload_bytes"].sum() / 1024
            fig.add_bar(x=g.index, y=g.values, name=d, marker_color=col)
        st.plotly_chart(style_fig(fig, 320, barmode="stack", title="Traffic per round (KB)", xaxis_title="Round",
                                  hovermode="closest"), width="stretch")
        with st.expander("Message log (what was actually transmitted)"):
            st.dataframe(msgs.drop(columns=["tensor_shapes"]), width="stretch", hide_index=True)
            st.caption(f"Tensor shapes in every message: {msgs['tensor_shapes'].iloc[0]}")
        with st.expander("Payload audit: searching every client upload for raw patient rows"):
            st.dataframe(pd.DataFrame(log["audits"]), width="stretch", hide_index=True)

    st.subheader("Limitations - what FL does *not* guarantee")
    st.markdown(
        """
<div class="note">Plain federated learning is <b>not</b> a formal privacy guarantee.</div>

- **Model updates leak information.** Gradient/weight inversion attacks can reconstruct training samples
  from updates, especially with small local datasets and few samples per batch. Membership-inference attacks
  can tell whether a patient was in a hospital's data.
- **The server sees each hospital's individual update** in plain FedAvg. **Secure aggregation** (cryptographic
  masking so the server only sees the sum) would hide individual updates.
- **Metadata leaks:** `n_k`, local loss and the federated standardization statistics (per-feature means and
  variances) reveal properties of each hospital's population, e.g. its malignancy rate.
- **Differential privacy** (e.g. DP-SGD or DP-FedAvg with clipping + calibrated noise) is required for a
  quantifiable bound (ε, δ) on what any single patient's record can influence.
- This is a **simulation in one process**; a real deployment also needs authenticated, encrypted transport
  (TLS), access control, auditing, and protection against malicious clients (poisoning, Byzantine updates).
""",
        unsafe_allow_html=True,
    )
    st.caption("The payload audit is a sanity demonstration that raw rows are not copied into messages, not a "
               "privacy proof.")


PAGES = {
    "Project overview": page_overview,
    "Training": page_training,
    "Results": page_results,
    "Hospital comparison": page_hospitals,
    "Privacy analysis": page_privacy,
}

with st.sidebar:
    st.markdown("### Federated Medical AI")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption(f"Flower FedAvg · PyTorch (CPU) · {CFG['partition']['num_clients']} hospitals · seed {CFG['seed']}")

PAGES[page]()
