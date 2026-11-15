from __future__ import annotations

from typing import Iterable, List, Optional


WEAPON_SCORES = {
    "gun": 100,
    "knife": 80,
    "weapon": 80,
}


def compute_risk(
    crowd_count: int,
    weapon_labels: Iterable[str],
    suspicious: bool,
    feed_tampered: bool = False,
    feed_integrity_score: float = 1.0,
) -> dict:
    """
    Unified Risk Engine: Fuses physical crowd surveillance with feed integrity verification.
    """
    weapon_labels = list(weapon_labels)
    score = int(crowd_count) * 5
    reasons: List[str] = []

    if crowd_count > 15:
        reasons.append(f"High crowd density detected ({crowd_count} persons in frame)")
    elif crowd_count > 5:
        reasons.append(f"Moderate crowd presence ({crowd_count} persons)")

    for label in weapon_labels:
        w_score = WEAPON_SCORES.get(label, 70)
        score += w_score
        reasons.append(f"Active weapon threat detected: {label.upper()} (+{w_score} pts)")

    if suspicious:
        score += 25
        reasons.append("High-velocity suspicious trajectory observed on armed subject (+25 pts)")

    # Feed integrity penalty: if video feed shows synthetic or deepfake manipulation
    if feed_tampered:
        score = max(score, 95)
        reasons.append("CRITICAL: Video stream integrity failure — possible deepfake injection / camera tampering")

    if score >= 100 or feed_tampered:
        level = "CRITICAL" if (weapon_labels or feed_tampered) else "DANGER"
    elif score >= 50:
        level = "ALERT"
    elif score >= 20:
        level = "ELEVATED"
    else:
        level = "SAFE"

    if not reasons:
        reasons.append("Normal optical capture; no weapon or crowd anomalies observed")

    return {
        "score": min(100, score),
        "level": level,
        "weapon_count": len(weapon_labels),
        "suspicious": suspicious,
        "feed_tampered": feed_tampered,
        "feed_integrity_score": round(float(feed_integrity_score), 3),
        "reasons": reasons,
    }
