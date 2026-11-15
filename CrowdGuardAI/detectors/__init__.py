"""
Universal Synthetic Media Detection Ensemble.
Exports provenance, pixel forensics, dynamic calibration, and video temporal analysis.
"""
from .calibration import calibrate_score, load as load_calibration
from .pixel import (
    CommForDetector,
    BFreeDetector,
    CLIPDetector,
    estimate_image_quality
)
from .provenance import check_provenance
from .video import (
    sample_video_frames,
    sample_video_frames_with_timestamps,
    sample_video_bytes,
    sample_video_bytes_with_timestamps,
    batch_video_frames
)
from .temporal import temporal_features, temporal_features_bytes
from . import gend_client

__all__ = [
    "calibrate_score",
    "load_calibration",
    "CommForDetector",
    "BFreeDetector",
    "CLIPDetector",
    "estimate_image_quality",
    "check_provenance",
    "sample_video_frames",
    "sample_video_frames_with_timestamps",
    "sample_video_bytes",
    "sample_video_bytes_with_timestamps",
    "batch_video_frames",
    "temporal_features",
    "temporal_features_bytes",
    "gend_client",
]
