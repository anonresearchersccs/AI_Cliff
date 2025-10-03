# %% [markdown]
# # AI Diffusion Economics — v2 Reproducible Notebook (Python Script Version)
# Pipeline: Scaling regressions with controls → KM & Cox → Monte Carlo NPV & APC.
# All parameters/paths via `config_ai_diffusion.yaml`. Exports in `./artifacts`.

# %%
# 1) Environment & Config
import os, sys, json, math, warnings, pathlib, datetime, platform, shutil
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    import lifelines
    from lifelines import KaplanMeierFitter, CoxPHFitter
except Exception:
    lifelines = None

BASE_DIR = pathlib.Path(".").resolve()
ART_DIR  = pathlib.Path("./artifacts")
FIG_DIR  = ART_DIR / "figures"
TAB_DIR  = ART_DIR / "tables"
LOG_DIR  = ART_DIR / "logs"
MOD_DIR  = ART_DIR / "models"
for d in [ART_DIR, FIG_DIR, TAB_DIR, LOG_DIR, MOD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

# Minimal YAML loader (no external deps)
def load_yaml_simple(path):
    d = {}
    stack = [d]
    key_stack = []
    indent_stack = [0]
    def set_in_current(k, v): stack[-1][k] = v
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip() or raw.strip().startswith("#"): continue
            indent = len(raw) - len(raw.lstrip())
            while indent < indent_stack[-1]:
                stack.pop(); key_stack.pop(); indent_stack.pop()
            if ":" in raw:
                k, v = raw.strip().split(":", 1)
                k = k.strip(); v = v.strip()
                if not v:
                    new_d = {}; set_in_current(k, new_d)
                    stack.append(new_d); key_stack.append(k); indent_stack.append(indent + 2)
                else:
                    if v.lower() in ["true","false"]:
                        vv = v.lower() == "true"
                    else:
                        try:
                            vv = float(v) if (("." in v) or ("e" in v.lower())) else int(v)
                        except:
                            vv = v.strip().strip("'").strip('"')
                    set_in_current(k, vv)
    return d

CFG_PATH = pathlib.Path("./config_ai_diffusion.yaml")
if not CFG_PATH.exists():
    CFG_TEMPLATE = """# YAML config for AI Diffusion Economics — v2
meta:
  project_name: "AI Diffusion Economics v2"
  author: "P. Martí et al."
  seed: 42
  smoketest: false
paths:
  scaling_data: "./data/scaling_dataset.parquet"
  diffusion_data: "./data/diffusion_pairs.parquet"
exports:
  fig_dir: "./artifacts/figures"
  tab_dir: "./artifacts/tables"
  log_dir: "./artifacts/logs"
  model_dir: "./artifacts/models"
scaling_spec:
  dep: "params"
  compute_col: ["train_compute_flops","train_compute_floats","train_flops"]
  params_col: ["parameters","params","n_params"]
  cost_col: ["train_cost_usd","cost_usd"]
  year_col: ["year","release_year"]
  arch_col: ["architecture","arch","model_family"]
  tier_col: ["tier"]
  org_type_col: ["org_type","developer_type"]
  params_scale: 1e9
  compute_scale: 1.0
  cost_scale: 1.0
  form_params_vs_compute: "log_params ~ log_compute + C(year) + C(arch)"
  form_cost_vs_params:    "log_cost ~ log_params + C(year) + C(arch)"
  robust_se: "HC3"
diffusion_spec:
  innov_date_col: ["innov_date","date_innov","t0"]
  follower_date_col: ["follower_date","date_follow","t1"]
  tier_col: ["tier"]
  org_type_col: ["org_type"]
  openness_col: ["openness","is_open"]
  window_months: 60
  km_fig: "survival_km.png"
  cox_fig: "cox_diagnostics.png"
  km_table: "km_summary.csv"
  cox_table: "cox_summary.csv"
investment_spec:
  target_params_billion: 1000.0
  gamma_compute: 1.697
  gamma_cost: 0.702
  compute_at_1b_params: 1.0e20
  price_per_flop_now: 2.0e-10
  price_decline_rate_annual: 0.35
  revenue_lognorm_mu: 15.0
  revenue_lognorm_sigma: 1.0
  post_competition_drop: 0.7
  horizon_months: 72
  annual_discount_rate: 0.2
  n_sims: 10000
  max_apc_share: 0.9
  tol_apc: 1.0e-4
  npv_hist_fig: "npv_hist.png"
  npv_summary_table: "npv_summary.csv"
  tornado_fig: "tornado_sensitivity.png"
"""
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        f.write(CFG_TEMPLATE)

config = load_yaml_simple(str(CFG_PATH))
for k in ["fig_dir","tab_dir","log_dir","model_dir"]:
    pathlib.Path(config["exports"][k]).mkdir(parents=True, exist_ok=True)

SESSION_INFO = {
    "timestamp": datetime.datetime.now().isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": getattr(np,"__version__",None),
    "pandas": getattr(pd,"__version__",None),
    "statsmodels": getattr(sm,"__version__",None),
    "lifelines": getattr(lifelines,"__version__",None) if lifelines else None,
}
with open(os.path.join(config["exports"]["log_dir"], "session_info.json"), "w", encoding="utf-8") as f:
    json.dump(SESSION_INFO, f, indent=2)
log("Environment ready and config loaded."); print(json.dumps(SESSION_INFO, indent=2))

# %%
# 2) Helpers & Harmonization
import re

def _first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def harmonize_scaling_columns(df: pd.DataFrame, cfg: dict):
    sc = cfg["scaling_spec"]
    col_map = {}
    col_map["params"] = _first_present(df, sc["params_col"]) or "parameters"
    col_map["compute"] = _first_present(df, sc["compute_col"]) or "train_compute_flops"
    col_map["cost"] = _first_present(df, sc["cost_col"]) or "train_cost_usd"
    col_map["year"] = _first_present(df, sc["year_col"]) or "year"
    col_map["arch"] = _first_present(df, sc["arch_col"]) or "arch"
    col_map["tier"] = _first_present(df, sc["tier_col"]) or "tier"
    col_map["org_type"] = _first_present(df, sc["org_type_col"]) or "org_type"

    out = df.copy()
    for k, v in col_map.items():
        if v not in out.columns:
            out[v] = np.nan
    out = out.rename(columns={col_map["params"]:"parameters",
                              col_map["compute"]:"compute",
                              col_map["cost"]:"cost",
                              col_map["year"]:"year",
                              col_map["arch"]:"arch",
                              col_map["tier"]:"tier",
                              col_map["org_type"]:"org_type"})
    out["parameters_b"] = out["parameters"].astype(float) / float(cfg["scaling_spec"]["params_scale"])
    out["compute_s"] = out["compute"].astype(float) / float(cfg["scaling_spec"]["compute_scale"])
    out["cost_s"] = out["cost"].astype(float) / float(cfg["scaling_spec"]["cost_scale"])

    for c in ["parameters_b","compute_s","cost_s"]:
        out = out.loc[out[c] > 0]
    out["log_params"] = np.log(out["parameters_b"])
    out["log_compute"] = np.log(out["compute_s"])
    out["log_cost"] = np.log(out["cost_s"])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["arch"] = out["arch"].astype("category")
    return out

def harmonize_diffusion_columns(df: pd.DataFrame, cfg: dict):
    ds = cfg["diffusion_spec"]
    innov = _first_present(df, ds["innov_date_col"]) or "innov_date"
    foll  = _first_present(df, ds["follower_date_col"]) or "follower_date"
    tier  = _first_present(df, ds["tier_col"]) or "tier"
    org   = _first_present(df, ds["org_type_col"]) or "org_type"
    open_ = _first_present(df, ds["openness_col"]) or "openness"

    out = df.copy()
    out[innov] = pd.to_datetime(out[innov], errors="coerce")
    out[foll]  = pd.to_datetime(out[foll], errors="coerce")
    out = out.rename(columns={innov:"innov_date", foll:"follower_date",
                              tier:"tier", org:"org_type", open_:"openness"})
    out["event"] = (~out["follower_date"].isna()).astype(int)
    out["lead_months"] = (out["follower_date"] - out["innov_date"]).dt.days/30.44
    return out

def save_table(df: pd.DataFrame, name: str, cfg: dict):
    out_csv = os.path.join(cfg["exports"]["tab_dir"], name if name.endswith(".csv") else name + ".csv")
    df.to_csv(out_csv, index=False)
    log(f"Table saved: {out_csv}")
    return out_csv

def save_fig(fig, name: str, cfg: dict):
    out_png = os.path.join(cfg["exports"]["fig_dir"], name if name.endswith(".png") else name + ".png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    log(f"Figure saved: {out_png}")
    return out_png

def tidy_ols_results(model):
    try:
        res = model.get_robustcov_results()
    except Exception:
        res = model
    summary = []
    ci = res.conf_int()
    for i, p in enumerate(res.params.index):
        summary.append({
            "term": p,
            "coef": float(res.params.iloc[i]),
            "se": float(res.bse.iloc[i]),
            "t": float(res.tvalues.iloc[i]),
            "p": float(res.pvalues.iloc[i]),
            "ci_low": float(ci.iloc[i,0]),
            "ci_high": float(ci.iloc[i,1]),
        })
    return pd.DataFrame(summary)

# %%
# 3) Data Loading & Smoketest
cfg = config
scaling_df = None; diffusion_df = None

sc_path = cfg["paths"]["scaling_data"]
if os.path.exists(sc_path):
    log(f"Loading scaling data from {sc_path}")
    raw_sc = pd.read_parquet(sc_path) if sc_path.endswith(".parquet") else pd.read_csv(sc_path)
    scaling_df = harmonize_scaling_columns(raw_sc, cfg)
else:
    log(f"Scaling data not found at {sc_path}.")

df_path = cfg["paths"]["diffusion_data"]
if os.path.exists(df_path):
    log(f"Loading diffusion data from {df_path}")
    raw_df = pd.read_parquet(df_path) if df_path.endswith(".parquet") else pd.read_csv(df_path)
    diffusion_df = harmonize_diffusion_columns(raw_df, cfg)
else:
    log(f"Diffusion data not found at {df_path}.")

if (scaling_df is None or diffusion_df is None) and bool(cfg["meta"].get("smoketest", False)):
    log("Smoketest enabled: generating synthetic data.")
    n = 200
    years = np.random.choice(range(2014, 2025), size=n)
    archs = np.random.choice(["Transformer","Mixture","Other"], size=n, p=[0.6,0.25,0.15])
    compute = np.exp(np.random.normal(45, 1.0, size=n))
    gamma_true = 1.65; alpha_true = -30.0
    params_b = np.exp(alpha_true + gamma_true*np.log(compute)) * 1e-9
    cost = compute * 2e-10
    scaling_df = pd.DataFrame({
        "parameters": (params_b * 1e9),
        "compute": compute,
        "cost": cost,
        "year": years,
        "arch": archs
    })
    scaling_df = harmonize_scaling_columns(scaling_df, cfg)

    m = 500
    start_dates = pd.to_datetime("2018-01-01") + pd.to_timedelta(np.random.randint(0, 2000, size=m), unit="D")
    te_months = np.maximum(1, np.random.lognormal(mean=2.0, sigma=0.6, size=m))
    follower_dates = start_dates + pd.to_timedelta((te_months*30.44).astype(int), unit="D")
    openness = np.random.choice([0,1], size=m, p=[0.6,0.4])
    tier = np.random.choice(["I","II","III"], size=m, p=[0.5,0.35,0.15])
    org_type = np.random.choice(["BigTech","Academic","Startup"], size=m, p=[0.5,0.2,0.3])
    diffusion_df = pd.DataFrame({
        "innov_date": start_dates,
        "follower_date": follower_dates,
        "openness": openness,
        "tier": tier,
        "org_type": org_type
    })
    diffusion_df = harmonize_diffusion_columns(diffusion_df, cfg)

if scaling_df is not None:
    log(f"Scaling data shape: {scaling_df.shape}")
else:
    log("Scaling data unavailable.")
if diffusion_df is not None:
    log(f"Diffusion data shape: {diffusion_df.shape}")
else:
    log("Diffusion data unavailable.")

# %%
# 4) Scaling regressions with controls
if scaling_df is None:
    log("No scaling data; skipping.")
else:
    sc = cfg["scaling_spec"]
    form_pc = sc["form_params_vs_compute"]
    form_cp = sc["form_cost_vs_params"]

    model_pc = smf.ols(formula=form_pc, data=scaling_df).fit(cov_type=sc.get("robust_se","HC3"))
    model_cp = smf.ols(formula=form_cp, data=scaling_df).fit(cov_type=sc.get("robust_se","HC3"))

    res_pc = tidy_ols_results(model_pc); res_cp = tidy_ols_results(model_cp)
    save_table(res_pc, "scaling_params_vs_compute.csv", cfg)
    save_table(res_cp, "scaling_cost_vs_params.csv", cfg)

    gamma_compute = float(model_pc.params.get("log_compute", float("nan")))
    gamma_cost    = float(model_cp.params.get("log_params", float("nan")))
    gamma_df = pd.DataFrame([{"metric":"gamma_compute","estimate":gamma_compute},
                             {"metric":"gamma_cost","estimate":gamma_cost}])
    save_table(gamma_df, "scaling_exponents_summary.csv", cfg)

    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.scatter(scaling_df["compute_s"], scaling_df["parameters_b"], s=8, alpha=0.6)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Training compute (scaled)"); ax1.set_ylabel("Parameters (billions)")
    ax1.set_title("Params vs Compute (log–log)")
    save_fig(fig1, "scaling_params_vs_compute.png", cfg); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.scatter(scaling_df["parameters_b"], scaling_df["cost_s"], s=8, alpha=0.6)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Parameters (billions)"); ax2.set_ylabel("Training cost (scaled USD)")
    ax2.set_title("Cost vs Params (log–log)")
    save_fig(fig2, "scaling_cost_vs_params.png", cfg); plt.close(fig2)

    log(model_pc.summary()); log(model_cp.summary())

# %%
# 5) Diffusion: KM and Cox PH
if diffusion_df is None:
    log("No diffusion data; skipping survival analysis.")
elif lifelines is None:
    log("lifelines not installed; install it and rerun for KM/Cox.")
else:
    dd = diffusion_df.dropna(subset=["innov_date"]).copy()
    if not bool(cfg["meta"].get("smoketest", False)):
        dd = dd.dropna(subset=["follower_date"])
        dd["duration"] = dd["lead_months"]
        dd["event_observed"] = 1
    else:
        cutoff = 24.0
        cens = dd["follower_date"].isna()
        dd["duration"] = dd["lead_months"]
        dd.loc[cens, "duration"] = cutoff
        dd["event_observed"] = (~cens).astype(int)

    kmf = KaplanMeierFitter()
    kmf.fit(dd["duration"].values, event_observed=dd["event_observed"].values, label="KM overall")
    median_te = kmf.median_survival_time_
    log(f"KM median TE (months): {median_te:.2f}")

    fig_km, ax = plt.subplots(figsize=(6,4))
    kmf.plot(ax=ax); ax.set_xlabel("Months"); ax.set_ylabel("Survival S(t)"); ax.set_title("Kaplan–Meier: Time to Follower")
    save_fig(fig_km, cfg["diffusion_spec"]["km_fig"], cfg); plt.close(fig_km)

    km_table = kmf.survival_function_.reset_index()
    km_table.columns = ["timeline_months","S"]
    save_table(km_table, cfg["diffusion_spec"]["km_table"], cfg)

    covars = []
    if "tier" in dd.columns: covars.append("tier")
    if "org_type" in dd.columns: covars.append("org_type")
    if "openness" in dd.columns: covars.append("openness")
    X = dd[["duration","event_observed"]].copy()
    for c in covars:
        if c == "openness":
            X[c] = pd.to_numeric(dd[c], errors="coerce").fillna(0).astype(int)
        else:
            dums = pd.get_dummies(dd[c].astype("category"), prefix=c, drop_first=True)
            X = pd.concat([X, dums], axis=1)
    cph = CoxPHFitter(); cph.fit(X, duration_col="duration", event_col="event_observed")
    cph_sum = cph.summary.reset_index().rename(columns={"index":"term"})
    save_table(cph_sum, cfg["diffusion_spec"]["cox_table"], cfg)
    diag_path = os.path.join(cfg["exports"]["log_dir"], "cox_check_assumptions.txt")
    with open(diag_path, "w", encoding="utf-8") as f:
        try:
            cph.check_assumptions(X, show_plots=False, p_value_threshold=0.05)
            f.write("Proportional hazards check completed.\n")
        except Exception as e:
            f.write(f"check_assumptions error: {e}\n")
    log(f"Cox PH diagnostics saved to: {diag_path}")

# %%
# 6) NPV Monte Carlo & APC
def compute_cost_from_params(params_b, cfg, gamma_compute=None):
    inv = cfg["investment_spec"]
    if gamma_compute is None or (isinstance(gamma_compute,float) and (np.isnan(gamma_compute))):
        gamma_compute = float(inv["gamma_compute"])
    compute_at_1b = float(inv["compute_at_1b_params"])
    compute = compute_at_1b * (params_b / 1.0) ** (1.0 / gamma_compute)
    price_flop_now = float(inv["price_per_flop_now"])
    cost0 = compute * price_flop_now
    return cost0, compute

def draw_te_months(cfg, km_table=None, n=1):
    if km_table is not None and len(km_table) > 2:
        tt = km_table["timeline_months"].values
        S  = km_table["S"].values
        F  = 1.0 - S
        F = F / F[-1] if F[-1] > 0 else F
        u = np.random.rand(n)
        draws = np.interp(u, F, tt)
        return draws
    else:
        med = 8.0; sigma_te = 0.6; mu_te = np.log(med)
        return np.random.lognormal(mean=mu_te, sigma=sigma_te, size=n)

def simulate_npv(cfg, gamma_compute=None, km_table=None, apc_share=0.0):
    inv = cfg["investment_spec"]
    params_b = float(inv["target_params_billion"])
    n_sims = int(inv["n_sims"])
    annual_r = float(inv["annual_discount_rate"])
    horizon = int(inv["horizon_months"])
    drop = float(inv["post_competition_drop"])
    mu_r = float(inv["revenue_lognorm_mu"])
    sig_r = float(inv["revenue_lognorm_sigma"])
    cost0, compute0 = compute_cost_from_params(params_b, cfg, gamma_compute=gamma_compute)
    net_cost0 = cost0 * (1.0 - apc_share)
    km_df = km_table if isinstance(km_table, pd.DataFrame) else None
    te_draws = draw_te_months(cfg, km_table=km_df, n=n_sims)
    rm = (1.0 + annual_r) ** (1.0/12.0) - 1.0
    npvs = np.zeros(n_sims)
    for i in range(n_sims):
        R0 = np.random.lognormal(mean=mu_r, sigma=sig_r)
        TE = max(1.0, te_draws[i])
        cash = 0.0
        for t in range(1, horizon+1):
            Rt = R0 if t <= TE else R0*(1.0 - drop)
            disc = (1.0 + rm) ** t
            cash += Rt / disc
        npvs[i] = -net_cost0 + cash
    return {"npvs": npvs, "cost0": cost0, "net_cost0": net_cost0, "compute0": compute0,
            "te_median_drawn": float(np.median(te_draws))}

def summarize_npv(npv_out, cfg):
    arr = npv_out["npvs"]
    df = pd.DataFrame([{
        "E_NPV": float(np.mean(arr)),
        "Std_NPV": float(np.std(arr, ddof=1)),
        "Prob_NPV_Pos": float(np.mean(arr>0.0)),
        "VaR_5pct": float(np.quantile(arr, 0.05)),
        "Cost0": float(npv_out["cost0"]),
        "NetCost0": float(npv_out["net_cost0"]),
        "Compute0": float(npv_out["compute0"]),
        "TE_median_drawn": float(npv_out["te_median_drawn"]),
    }])
    save_table(df, cfg["investment_spec"]["npv_summary_table"], cfg)
    return df

def plot_npv_hist(npv_out, cfg):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(npv_out["npvs"], bins=50, alpha=0.8)
    ax.set_xlabel("NPV (USD)"); ax.set_ylabel("Frequency"); ax.set_title("Monte Carlo NPV")
    save_fig(fig, cfg["investment_spec"]["npv_hist_fig"], cfg); plt.close(fig)

def tornado_sensitivity(cfg, base_gamma, km_table=None):
    base = simulate_npv(cfg, gamma_compute=base_gamma, km_table=km_table, apc_share=0.0)
    base_E = np.mean(base["npvs"])
    scenarios = {"gamma_compute +10%": base_gamma*1.1, "gamma_compute -10%": base_gamma*0.9}
    labels, vals = [], []
    for name, g in scenarios.items():
        out = simulate_npv(cfg, gamma_compute=g, km_table=km_table, apc_share=0.0)
        labels.append(name); vals.append(np.mean(out["npvs"]) - base_E)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(labels, vals)
    ax.set_xlabel("Δ E[NPV] vs base (USD)"); ax.set_title("One-at-a-time sensitivity")
    save_fig(fig, cfg["investment_spec"]["tornado_fig"], cfg); plt.close(fig)

def find_apc_star(cfg, base_gamma, km_table=None):
    lo, hi = 0.0, float(cfg["investment_spec"]["max_apc_share"])
    tol = float(cfg["investment_spec"]["tol_apc"])
    def E_at(apc): return float(np.mean(simulate_npv(cfg, gamma_compute=base_gamma, km_table=km_table, apc_share=apc)["npvs"]))
    for _ in range(40):
        mid = 0.5*(lo+hi); Emid = E_at(mid)
        if abs(Emid) < 1e-3: return mid
        if Emid < 0: lo = mid
        else: hi = mid
        if hi - lo < tol: break
    return 0.5*(lo+hi)

gamma_path = os.path.join(cfg["exports"]["tab_dir"], "scaling_exponents_summary.csv")
if os.path.exists(gamma_path):
    gdf = pd.read_csv(gamma_path)
    base_gamma = float(gdf.loc[gdf["metric"]=="gamma_compute","estimate"].iloc[0])
    log(f"Using regression-estimated gamma_compute={base_gamma:.3f}")
else:
    base_gamma = float(cfg["investment_spec"]["gamma_compute"])
    log(f"Using config gamma_compute={base_gamma:.3f}")

km_path = os.path.join(cfg["exports"]["tab_dir"], cfg["diffusion_spec"]["km_table"])
km_df = pd.read_csv(km_path) if os.path.exists(km_path) else None

mc_out = simulate_npv(cfg, gamma_compute=base_gamma, km_table=km_df, apc_share=0.0)
mc_sum = summarize_npv(mc_out, cfg); print(mc_sum)
plot_npv_hist(mc_out, cfg)

tornado_sensitivity(cfg, base_gamma, km_table=km_df)
apc_star = find_apc_star(cfg, base_gamma, km_table=km_df)
with open(os.path.join(cfg["exports"]["tab_dir"], "apc_star.txt"), "w", encoding="utf-8") as f:
    f.write(f"{apc_star:.6f}\n")
log(f"APC* (share of upfront cost) for E[NPV] >= 0: {apc_star:.4f}")

# %%
# 7) Exports & Session Snapshot
snap = os.path.join(cfg["exports"]["log_dir"], f"config_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
shutil.copyfile("config_ai_diffusion.yaml", snap)
log(f"Config snapshot saved: {snap}")
log("Run complete. Check ./artifacts for figures, tables, logs.")
