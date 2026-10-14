"""Dump every paper-cited result value from the (n=6) JSONs into a manifest for the Phase-4 rewrite."""
import json
from pathlib import Path
G = Path(__file__).resolve().parent.parent.parent / "runs/protocol_v2/gen"


def L(p):
    return json.load(open(G / p))


def m(d, k):
    return d[k]["mean"]


def s(d, k):
    return d[k]["std"]


out = []
def P(x): out.append(x)

# ---- headline single (unsup_robust) ----
for w in ["for", "biodiv"]:
    j = L(f"unsup_robust_{w}.json")
    gd = j["geo_dim"]
    P(f"\n## unsup_robust_{w}  K={j['K']} n={j['n']} geo_dim={gd}")
    c5 = j["CLIP_512"]; cm = j[f"CLIP_matched_{gd}d"]; ge = j[f"geo_{gd}d"]
    for nm, a in [("CLIP_512", c5), (f"CLIP_{gd}d", cm), (f"geo_{gd}d", ge)]:
        P(f"  {nm}: NMI {m(a,'NMI'):.4f}±{s(a,'NMI'):.4f}  F1 {m(a,'macroF1'):.4f}±{s(a,'macroF1'):.4f}  OA {m(a,'OA'):.4f}  mIoU {m(a,'mIoU'):.4f}")
    d5 = j["delta_512_minus_geo"]; dm = j["delta_matched_minus_geo"]
    P(f"  delta512-geo: " + ", ".join(f"{k} {d5[k]['mean']:+.4f}(fp{d5[k]['frac_positive']})" for k in ["NMI","macroF1","OA","mIoU"]))
    P(f"  deltaMatched-geo: " + ", ".join(f"{k} {dm[k]['mean']:+.4f}(fp{dm[k]['frac_positive']})" for k in ["NMI","macroF1","OA","mIoU"]))
    P(f"  ratio CLIP512/geo NMI = {m(c5,'NMI')/m(ge,'NMI'):.2f}x ; CLIPmatched/geo = {m(cm,'NMI')/m(ge,'NMI'):.2f}x")

# ---- multi backbone ----
for w in ["for", "biodiv"]:
    j = L(f"unsup_robust_multi_{w}.json")
    gd = j["geo_dim"]
    P(f"\n## unsup_robust_multi_{w}  K={j['K']} n={j['n']}")
    for nm in ["CLIP","DINOv2","ImageNet","random"]:
        nat = j[f"{nm}_native"]; mat = j[f"{nm}_matched_{gd}d"]; dl = j[f"delta_{nm}_matched_minus_geo"]
        P(f"  {nm}: native NMI {m(nat,'NMI'):.4f} | matched{gd}d NMI {m(mat,'NMI'):.4f} F1 {m(mat,'macroF1'):.4f} OA {m(mat,'OA'):.4f} mIoU {m(mat,'mIoU'):.4f} | dNMI {dl['NMI']['mean']:+.4f}(fp{dl['NMI']['frac_positive']})")
    ge = j[f"geo_{gd}d"]; P(f"  geo{gd}d NMI {m(ge,'NMI'):.4f} F1 {m(ge,'macroF1'):.4f} OA {m(ge,'OA'):.4f} mIoU {m(ge,'mIoU'):.4f}")

# ---- view ablation ----
for w in ["for","biodiv"]:
    j = L(f"view_ablation_{w}.json")
    P(f"\n## view_ablation_{w}  K={j['K']} n={j['n']} geoNMI {j['geo_matched']['NMI']['mean']:.4f} subset {j['subset']}")
    for n in j["nlist"]:
        pn=j["per_n"][str(n)]; P(f"  n={n}: NMI {pn['NMI']['mean']:.4f} F1 {pn['macroF1']['mean']:.4f} OA {pn['OA']['mean']:.4f} mIoU {pn['mIoU']['mean']:.4f}")

# ---- dim sweep ----
for w in ["for","biodiv"]:
    j = L(f"dim_sweep_{w}.json")
    P(f"\n## dim_sweep_{w} keys={list(j.keys())[:3]}...")
    P("  " + json.dumps(j)[:600])

# ---- clustering methods (CLIP + dino) ----
for tag in ["", "dino_"]:
    for w in ["for","biodiv"]:
        j = L(f"clustering_methods_{tag}{w}.json")
        P(f"\n## clustering_methods_{tag}{w}")
        P("  " + json.dumps(j)[:700])

# ---- k sensitivity ----
for w in ["for","biodiv"]:
    j = L(f"k_sensitivity_{w}.json"); P(f"\n## k_sensitivity_{w}"); P("  " + json.dumps(j)[:500])

# ---- permutation ----
for f in ["perm_test.json","perm_test_dino.json","perm_test_multi_for.json","perm_test_multi_biodiv.json"]:
    j=L(f); P(f"\n## {f}"); P("  "+json.dumps(j)[:600])

# ---- source control ----
for f in ["source_control.json","source_control_robust.json","source_control_robust_dino.json","source_control_multi_for.json","source_control_multi_biodiv.json"]:
    j=L(f); P(f"\n## {f}"); P("  "+json.dumps(j)[:700])

# ---- scores ----
for w in ["for","biodiv"]:
    j=L(f"score_{w}.json")
    P(f"\n## score_{w}: CLIP_lin F1 {j['CLIP_linear']['pooled_f1']} best_geo {j['best_geo']} {j[j['best_geo']]['pooled_f1']} delta {j['delta_clip_minus_bestgeo']} p {j.get('wilcoxon_p')} bonf3 {j.get('p_bonferroni_x3')} verdict {j['verdict']} dominant {j['dominant_frac']} CLIP_pca {j['CLIP_pca_matched']['pooled_f1']} delta_pca {j['delta_clip_pca_minus_bestgeo']}")

# ---- supervised matched sf, rev, random baseline ----
P("\n## supervised_matched_sf"); P("  "+json.dumps(L("supervised_matched_sf.json")))
for w in ["for","biodiv"]:
    j=L(f"rev_predict_source_{w}.json"); P(f"\n## rev_predict_source_{w}"); P("  "+json.dumps(j)[:500])
j=L("rev_perclass_for.json"); P(f"\n## rev_perclass_for corr {j['corr_share_delta']} high5 {j['mean_delta_high_confound']} low {j['mean_delta_low_confound']}")
P("\n## random_nmi_baseline"); P("  "+json.dumps(L("random_nmi_baseline.json"))[:300])

print("\n".join(out))
