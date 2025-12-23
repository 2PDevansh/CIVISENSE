def compute_drift(baseline_conf, current_conf):
    drift = abs(current_conf - baseline_conf)

    if drift < 0.1:
        status = "STABLE"
    elif drift < 0.2:
        status = "WARNING"
    else:
        status = "RETRAIN_SUGGESTED"

    return round(drift, 3), status
