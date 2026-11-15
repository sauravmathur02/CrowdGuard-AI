from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from crowd.behavior import BehaviorAnalyzer
from crowd.risk_engine import compute_risk
from crowd.smoothing import DetectionSmoother
from crowd.weapon_classifier import WeaponClassifier
from crowd.yolo_detector import Detection, YOLODetector


@dataclass
class ProcessedFrame:
    source_id: str
    timestamp: str
    detections: List[Detection]
    crowd_count: int
    risk: dict
    suspicious_ids: List[int]
    weapon_labels: List[str]
    annotated_frame: np.ndarray
    alert_active: bool
    detection_count: int
    active_tracks: int
    dominant_weapon: str
    feed_tampered: bool = False
    feed_integrity_score: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "crowd_count": self.crowd_count,
            "risk": self.risk,
            "alert_active": self.alert_active,
            "weapon_labels": self.weapon_labels,
            "suspicious_ids": self.suspicious_ids,
            "detection_count": self.detection_count,
            "active_tracks": self.active_tracks,
            "dominant_weapon": self.dominant_weapon,
            "feed_tampered": self.feed_tampered,
            "feed_integrity_score": self.feed_integrity_score,
            "detections": [d.to_dict() for d in self.detections],
        }

    def to_base64_jpg(self, quality: int = 80) -> str:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", self.annotated_frame, encode_param)
        return "data:image/jpeg;base64," + base64.b64encode(enc).decode("utf-8")


@dataclass(frozen=True)
class RenderOptions:
    show_boxes: bool = True
    show_labels: bool = True
    show_tracking: bool = True
    highlight_objects: bool = True
    show_hud: bool = True


class FeedProcessor:
    def __init__(
        self,
        source_id: str,
        detector: YOLODetector,
        classifier: WeaponClassifier,
        smoother: Optional[DetectionSmoother] = None,
        behavior: Optional[BehaviorAnalyzer] = None,
    ) -> None:
        self.source_id = source_id
        self.detector = detector
        self.classifier = classifier
        self.smoother = smoother or DetectionSmoother()
        self.behavior = behavior or BehaviorAnalyzer()
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    def reset(self) -> None:
        self.detector.reset_tracking()
        self.smoother.reset()
        self.behavior.reset()
        self.track_history.clear()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[str] = None,
        render_options: Optional[RenderOptions] = None,
        feed_tampered: bool = False,
        feed_integrity_score: float = 1.0,
    ) -> ProcessedFrame:
        timestamp = timestamp or datetime.now().astimezone().isoformat(timespec="seconds")
        render_options = render_options or RenderOptions()

        detections = self._classify_detections(frame_bgr)
        detections = self.smoother.update(detections)
        behavior_state = self.behavior.analyze(detections)

        for detection in detections:
            if detection.track_id is None:
                continue
            scale = behavior_state["confidence_scale"].get(detection.track_id, 1.0)
            detection.conf = min(1.0, detection.conf * scale)

        crowd_count = self.detector.unique_track_count(detections, "person")
        weapon_labels = [
            detection.weapon_type or detection.label
            for detection in detections
            if detection.label == "weapon"
        ]
        active_tracks = len({detection.track_id for detection in detections if detection.track_id is not None})
        risk = compute_risk(
            crowd_count,
            weapon_labels,
            behavior_state["suspicious"],
            feed_tampered=feed_tampered,
            feed_integrity_score=feed_integrity_score,
        )

        annotated_frame = draw_annotations(
            frame_bgr.copy(),
            detections,
            self.track_history,
            crowd_count,
            risk,
            behavior_state["suspicious_ids"],
            self.source_id,
            render_options,
            feed_integrity_score=feed_integrity_score,
            feed_tampered=feed_tampered,
        )

        return ProcessedFrame(
            source_id=self.source_id,
            timestamp=timestamp,
            detections=detections,
            crowd_count=crowd_count,
            risk=risk,
            suspicious_ids=sorted(behavior_state["suspicious_ids"]),
            weapon_labels=[w for w in weapon_labels if w],
            annotated_frame=annotated_frame,
            alert_active=bool(weapon_labels) or feed_tampered,
            detection_count=len(detections),
            active_tracks=active_tracks,
            dominant_weapon=weapon_labels[0] if weapon_labels else "None",
            feed_tampered=feed_tampered,
            feed_integrity_score=feed_integrity_score,
        )

    def _classify_detections(self, frame_bgr: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        for detection in self.detector.detect(frame_bgr):
            if detection.label != "weapon":
                detections.append(detection)
                continue

            if not self.classifier.available:
                detections.append(detection)
                continue

            weapon_type, classifier_conf = self.classifier.classify(frame_bgr, detection.xyxy)
            if classifier_conf < self.classifier.confidence_floor:
                continue

            detection.weapon_type = weapon_type
            detection.classifier_conf = classifier_conf
            detection.conf = min(1.0, (0.6 * detection.detector_conf) + (0.4 * classifier_conf))
            detections.append(detection)

        return detections


def draw_annotations(
    frame_bgr: np.ndarray,
    detections: List[Detection],
    track_history: Dict[int, List[Tuple[int, int]]],
    crowd_count: int,
    risk: dict,
    suspicious_ids: set,
    source_id: str,
    render_options: RenderOptions,
    feed_integrity_score: float = 1.0,
    feed_tampered: bool = False,
) -> np.ndarray:
    color_map = {
        "person": (40, 220, 110),     # Vibrant Green
        "gun": (40, 40, 240),         # Red
        "knife": (0, 150, 255),       # Orange
        "weapon": (40, 40, 240),
    }

    if not render_options.show_tracking:
        track_history.clear()

    # Draw trajectories
    if render_options.show_tracking:
        for detection in detections:
            if detection.label == "person" and detection.track_id is not None:
                x1, y1, x2, y2 = detection.xyxy
                center = (int((x1 + x2) / 2), int(y2))  # base of bounding box
                history = track_history[detection.track_id]
                history.append(center)
                if len(history) > 25:
                    history.pop(0)
                for idx in range(1, len(history)):
                    alpha = idx / len(history)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame_bgr, history[idx - 1], history[idx], (40, 220, 110), thickness)

    # Draw bounding boxes and labels
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        label = detection.display_label
        color = color_map.get(label, (0, 255, 0))
        if detection.track_id in suspicious_ids:
            color = (0, 0, 255)

        if render_options.show_boxes:
            if render_options.highlight_objects:
                overlay = frame_bgr.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.12, frame_bgr, 0.88, 0, frame_bgr)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        if render_options.show_labels:
            tag = f"{label.upper()} {detection.conf:.2f}"
            if render_options.show_tracking and detection.track_id is not None:
                tag += f" | ID #{detection.track_id}"
            if detection.smoothed:
                tag += " [HOLD]"

            # Label banner
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), (15, 23, 42), -1)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, 1)
            cv2.putText(
                frame_bgr,
                tag,
                (x1 + 5, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    active_ids = {detection.track_id for detection in detections if detection.track_id is not None}
    for track_id in list(track_history.keys()):
        if track_id not in active_ids:
            track_history.pop(track_id, None)

    # HUD Overlay (Cyber-Security Command Style)
    if render_options.show_hud:
        h, w = frame_bgr.shape[:2]
        hud_w = min(360, w - 20)
        hud_h = 135
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (10, 15, 26), -1)
        cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (12, 12), (12 + hud_w, 12 + hud_h), (56, 189, 248), 1)

        # HUD Text
        cv2.putText(frame_bgr, f"CROWDGUARD // {source_id.upper()}", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Risk indicator
        risk_level = risk.get("level", "SAFE")
        risk_color = (40, 220, 110) if risk_level == "SAFE" else ((0, 200, 255) if risk_level == "ALERT" else (40, 40, 255))
        cv2.putText(frame_bgr, f"THREAT LEVEL: {risk_level} ({risk.get('score', 0)})", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, risk_color, 2, cv2.LINE_AA)

        # Crowd count
        cv2.putText(frame_bgr, f"ACTIVE PERSON TRACKS: {crowd_count}", (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (226, 232, 240), 1, cv2.LINE_AA)

        # Feed Integrity Badge
        if feed_tampered:
            integ_text = f"FEED INTEGRITY: TAMPERED ({feed_integrity_score:.0%})"
            integ_color = (40, 40, 255)
        else:
            integ_text = f"FEED INTEGRITY: VERIFIED ({feed_integrity_score:.0%})"
            integ_color = (40, 220, 110)
        cv2.putText(frame_bgr, integ_text, (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.50, integ_color, 2, cv2.LINE_AA)

    return frame_bgr
