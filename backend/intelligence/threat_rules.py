def evaluate_prosody(prosody):
    """
    Rule-based acoustic analysis.

    Returns:
        score: 0-100 acoustic suspiciousness score
        reasons: detected acoustic indicators
    """

    score = 0
    reasons = []

    pitch_mean = prosody.get("pitch_mean", 0.0)
    pitch_variation = prosody.get("pitch_variation", 0.0)
    energy_mean = prosody.get("energy_mean", 0.0)
    energy_std = prosody.get("energy_std", 0.0)
    zcr = prosody.get("zero_crossing_rate", 0.0)

    # High fundamental frequency
    if pitch_mean > 350:
        score += 30
        reasons.append("Unusually high fundamental frequency")

    # Very high pitch variation
    if pitch_variation > 0.70:
        score += 50
        reasons.append("Extremely high pitch variation")

    elif pitch_variation > 0.40:
        score += 30
        reasons.append("High pitch variation")

    # Very low pitch variation
    elif pitch_variation < 0.04 and pitch_mean > 100:
        score += 15
        reasons.append("Highly uniform pitch pattern")

    # Energy variation
    if energy_mean > 0:
        energy_variation = energy_std / energy_mean

        if energy_variation < 0.20:
            score += 15
            reasons.append("Unusually uniform energy pattern")

    # High zero crossing rate
    if zcr > 0.15:
        score += 10
        reasons.append("High spectral activity")

    score = min(score, 100)

    if not reasons:
        reasons.append("No major acoustic indicators")

    return score, reasons