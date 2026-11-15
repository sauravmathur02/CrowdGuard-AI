# Crowd Intelligence Package for CrowdGuard AI
from crowd.yolo_detector import YOLODetector, Detection
from crowd.weapon_classifier import WeaponClassifier
from crowd.smoothing import DetectionSmoother
from crowd.behavior import BehaviorAnalyzer
from crowd.risk_engine import compute_risk
from crowd.feed_processor import FeedProcessor, ProcessedFrame, RenderOptions
