def compute_global_importance(model, feature_names):
    rf = None
    for s in model.stages:
        if hasattr(s, "featureImportances"):
            rf = s
            break

    if rf is None:
        return []

    fi = rf.featureImportances
    total = float(sum(fi))

    imps = []
    for i, name in enumerate(feature_names):
        val = float(fi[i])
        pct = (val / total * 100) if total > 0 else val * 100
        imps.append({"feature": name, "importance": round(pct, 4)})

    return sorted(imps, key=lambda x: x["importance"], reverse=True)
