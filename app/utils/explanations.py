def compute_global_importance(model, feature_names):
    # Handle sklearn model (dict with 'model' key)
    if isinstance(model, dict):
        sklearn_model = model.get('model')
        if sklearn_model and hasattr(sklearn_model, 'feature_importances_'):
            fi = sklearn_model.feature_importances_
            total = float(sum(fi))
            imps = []
            for i, name in enumerate(feature_names):
                val = float(fi[i])
                pct = (val / total * 100) if total > 0 else val * 100
                imps.append({"feature": name, "importance": round(pct, 4)})
            return sorted(imps, key=lambda x: x["importance"], reverse=True)
        return []
    
    # Handle Spark PipelineModel
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
