"""
Severity estimation engine for road damage detections.

Implements a transparent, heuristic-based visual severity estimator.
Severity is estimated from image-space characteristics — NOT from
physical/structural assessment.

IMPORTANT LIMITATION:
This is visual severity estimation based on bounding-box area relative
to image area, damage category, and detection confidence. It does NOT
determine actual physical dimensions (which would require camera
calibration and depth estimation) and does NOT assess structural road
safety. Professional road inspection is still required.

Severity levels:
    - Low: Small detected region, lower risk indicators
    - Medium: Moderate detected region
    - High: Large detected region, higher risk indicators
"""

from dataclasses import dataclass

from app.config.settings import get_settings
from app.cv.detector import Detection
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Damage category severity weights — potholes are visually
# more severe than linear cracks of the same relative size
DEFAULT_CATEGORY_WEIGHTS = {
    "D00": 0.8,   # Longitudinal Crack
    "D10": 0.8,   # Transverse Crack
    "D20": 1.0,   # Alligator Crack (network of cracks)
    "D40": 1.2,   # Pothole (typically more severe)
}


@dataclass
class SeverityResult:
    """Result of severity estimation for a detection."""
    level: str          # "Low", "Medium", "High"
    score: float        # Composite score (0.0 - 1.0)
    area_ratio: float   # bbox area / image area
    category_weight: float
    confidence_factor: float


class SeverityEngine:
    """
    Heuristic severity estimator based on visual image-space features.

    Uses configurable thresholds for area ratio, damage category
    weights, and confidence to assign Low/Medium/High severity.
    """

    def __init__(self):
        self.settings = get_settings()
        self.low_threshold = self.settings.severity_low_threshold
        self.medium_threshold = self.settings.severity_medium_threshold
        self.high_threshold = self.settings.severity_high_threshold
        self.category_weights = DEFAULT_CATEGORY_WEIGHTS.copy()

    def estimate(
        self,
        detection: Detection,
        image_width: int,
        image_height: int,
    ) -> SeverityResult:
        """
        Estimate the visual severity of a single detection.

        The composite severity score combines:
        1. Area ratio: bbox area / image area (primary factor)
        2. Category weight: damage type severity modifier
        3. Confidence factor: higher confidence slightly increases score

        Args:
            detection: The detection to evaluate.
            image_width: Width of the source image.
            image_height: Height of the source image.

        Returns:
            SeverityResult with level, score, and component factors.
        """
        image_area = image_width * image_height
        if image_area == 0:
            return SeverityResult(
                level="Low", score=0.0,
                area_ratio=0.0, category_weight=1.0,
                confidence_factor=0.0,
            )

        # Component 1: Area ratio
        area_ratio = detection.bbox_area / image_area

        # Component 2: Category weight
        category_weight = self.category_weights.get(
            detection.class_code, 1.0
        )

        # Component 3: Confidence factor (0.8 - 1.0 range)
        confidence_factor = 0.8 + 0.2 * detection.confidence

        # Composite score
        weighted_area = area_ratio * category_weight * confidence_factor
        # Normalize to 0-1 range using the high threshold as reference
        score = min(weighted_area / self.high_threshold, 1.0)

        # Determine level
        if weighted_area >= self.high_threshold:
            level = "High"
        elif weighted_area >= self.medium_threshold:
            level = "Medium"
        else:
            level = "Low"

        return SeverityResult(
            level=level,
            score=round(score, 4),
            area_ratio=round(area_ratio, 6),
            category_weight=category_weight,
            confidence_factor=round(confidence_factor, 4),
        )

    def estimate_batch(
        self,
        detections: list[Detection],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """
        Estimate severity for a batch of detections and update
        the severity field on each Detection in-place.

        Args:
            detections: List of Detection objects.
            image_width: Image width.
            image_height: Image height.

        Returns:
            The same list of detections with severity populated.
        """
        for det in detections:
            result = self.estimate(det, image_width, image_height)
            det.severity = result.level

        return detections
