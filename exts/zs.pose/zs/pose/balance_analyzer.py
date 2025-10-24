"""
Balance Analysis

Computes support polygon, center of mass, and balance metrics
for martial arts training.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import ConvexHull
import cv2

from .pose_estimator import PoseKeypoints


class BalanceMetrics:
    """Container for balance analysis results."""

    def __init__(self):
        """Initialize empty metrics."""
        self.support_polygon: Optional[np.ndarray] = None  # (N, 2) polygon vertices
        self.center_of_mass: Optional[Tuple[float, float]] = None  # (x, y)
        self.balance_score: float = 0.0  # 0-100
        self.com_inside_support: bool = False
        self.distance_to_edge: float = 0.0  # Normalized distance
        self.stance_width: float = 0.0  # Normalized width
        self.stance_type: str = "unknown"  # "parallel", "staggered", "single_leg"


def compute_support_polygon(keypoints: PoseKeypoints) -> Optional[np.ndarray]:
    """
    Compute support polygon from foot keypoints.

    The support polygon is the convex hull of all foot contact points.

    Args:
        keypoints: Pose keypoints

    Returns:
        Array of polygon vertices (N, 2), or None if insufficient points
    """
    # Collect foot points
    foot_points = []

    foot_keypoint_names = [
        'left_ankle',
        'right_ankle',
        'left_heel',
        'right_heel',
        'left_foot',
        'right_foot'
    ]

    for name in foot_keypoint_names:
        if keypoints.is_visible(name, threshold=0.3):
            pt = keypoints.get_2d(name)
            if pt is not None:
                foot_points.append(pt)

    if len(foot_points) < 3:
        # Need at least 3 points for convex hull
        return None

    points_array = np.array(foot_points)

    # Compute convex hull
    try:
        hull = ConvexHull(points_array)
        polygon = points_array[hull.vertices]
        return polygon
    except Exception as e:
        return None


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Check if point is inside polygon.

    Uses ray casting algorithm.

    Args:
        point: (x, y) point to test
        polygon: (N, 2) polygon vertices

    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]

        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x, p1y = p2x, p2y

    return inside


def distance_to_polygon_edge(
    point: Tuple[float, float],
    polygon: np.ndarray
) -> float:
    """
    Compute minimum distance from point to polygon edge.

    Args:
        point: (x, y) point
        polygon: (N, 2) polygon vertices

    Returns:
        Minimum distance to any edge
    """
    min_dist = float('inf')
    px, py = point

    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # Distance from point to line segment
        dist = point_to_segment_distance((px, py), p1, p2)
        min_dist = min(min_dist, dist)

    return min_dist


def point_to_segment_distance(
    point: Tuple[float, float],
    seg_start: np.ndarray,
    seg_end: np.ndarray
) -> float:
    """
    Distance from point to line segment.

    Args:
        point: (x, y) point
        seg_start: Segment start point
        seg_end: Segment end point

    Returns:
        Distance
    """
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    # Vector from start to end
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        # Segment is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Parameter t for closest point on segment
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = np.clip(t, 0, 1)

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance
    dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    return dist


def classify_stance(keypoints: PoseKeypoints) -> Tuple[str, float]:
    """
    Classify stance type and compute width.

    Args:
        keypoints: Pose keypoints

    Returns:
        (stance_type, stance_width)
    """
    left_ankle = keypoints.get_2d('left_ankle')
    right_ankle = keypoints.get_2d('right_ankle')

    if left_ankle is None or right_ankle is None:
        return "unknown", 0.0

    # Compute ankle distance
    dist_x = abs(left_ankle[0] - right_ankle[0])
    dist_y = abs(left_ankle[1] - right_ankle[1])

    # Check visibility
    left_vis = keypoints.get_visibility('left_ankle')
    right_vis = keypoints.get_visibility('right_ankle')

    if left_vis < 0.3 or right_vis < 0.3:
        return "single_leg", 0.0

    # Classify based on foot positioning
    if dist_y > dist_x * 1.5:
        # Feet are more front-back than side-side
        stance_type = "staggered"
    else:
        # Feet are more side-by-side
        stance_type = "parallel"

    # Stance width (horizontal distance)
    stance_width = dist_x

    return stance_type, stance_width


class BalanceAnalyzer:
    """
    Analyzes balance from pose keypoints.
    """

    def __init__(self):
        """Initialize balance analyzer."""
        self.frame_history: List[BalanceMetrics] = []
        self.history_length = 30  # Keep last 30 frames

    def analyze(self, keypoints: PoseKeypoints) -> BalanceMetrics:
        """
        Analyze balance from pose keypoints.

        Args:
            keypoints: Pose keypoints

        Returns:
            Balance metrics
        """
        metrics = BalanceMetrics()

        # Compute support polygon
        polygon = compute_support_polygon(keypoints)
        metrics.support_polygon = polygon

        # Get center of mass
        com = keypoints.get_center_of_mass()
        metrics.center_of_mass = com

        # Classify stance
        stance_type, stance_width = classify_stance(keypoints)
        metrics.stance_type = stance_type
        metrics.stance_width = stance_width

        if polygon is not None and com is not None:
            # Check if COM is inside support polygon
            metrics.com_inside_support = point_in_polygon(com, polygon)

            # Distance to edge
            dist = distance_to_polygon_edge(com, polygon)
            metrics.distance_to_edge = dist

            # Balance score (0-100)
            # Higher score = more stable (COM further from edge)
            # Normalize by stance width
            if stance_width > 0:
                normalized_dist = dist / stance_width
            else:
                normalized_dist = dist

            if metrics.com_inside_support:
                # Inside: score based on distance to edge
                # 0.5 * stance_width = score 100
                # 0.0 * stance_width = score 50
                score = min(100, 50 + normalized_dist * 100)
            else:
                # Outside: low score based on how far outside
                score = max(0, 50 - normalized_dist * 100)

            metrics.balance_score = score
        else:
            metrics.balance_score = 0.0

        # Add to history
        self.frame_history.append(metrics)
        if len(self.frame_history) > self.history_length:
            self.frame_history.pop(0)

        return metrics

    def get_average_balance_score(self, frames: int = 30) -> float:
        """Get average balance score over recent frames."""
        if not self.frame_history:
            return 0.0

        recent = self.frame_history[-frames:]
        scores = [m.balance_score for m in recent]

        return np.mean(scores)

    def detect_balance_edge_alert(self, threshold: float = 30.0) -> bool:
        """
        Detect if athlete is close to losing balance.

        Args:
            threshold: Balance score threshold for alert

        Returns:
            True if alert should be raised
        """
        if len(self.frame_history) < 5:
            return False

        # Check recent frames
        recent_scores = [m.balance_score for m in self.frame_history[-5:]]

        # Alert if consistently low or dropping
        if np.mean(recent_scores) < threshold:
            return True

        # Alert if rapid drop
        if len(recent_scores) >= 2:
            drop = recent_scores[0] - recent_scores[-1]
            if drop > 20:
                return True

        return False

    def draw_balance_overlay(
        self,
        frame: np.ndarray,
        metrics: BalanceMetrics
    ) -> np.ndarray:
        """
        Draw balance metrics overlay on frame.

        Args:
            frame: Input frame
            metrics: Balance metrics

        Returns:
            Frame with overlay
        """
        output = frame.copy()
        H, W = frame.shape[:2]

        # Draw support polygon
        if metrics.support_polygon is not None:
            polygon_px = (metrics.support_polygon * [W, H]).astype(np.int32)
            cv2.polylines(output, [polygon_px], True, (0, 255, 255), 2)

        # Draw center of mass
        if metrics.center_of_mass is not None:
            com_px = (
                int(metrics.center_of_mass[0] * W),
                int(metrics.center_of_mass[1] * H)
            )
            color = (0, 255, 0) if metrics.com_inside_support else (0, 0, 255)
            cv2.circle(output, com_px, 8, color, -1)
            cv2.putText(
                output, "COM", (com_px[0] + 10, com_px[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Draw balance score
        score_text = f"Balance: {metrics.balance_score:.0f}"
        cv2.putText(
            output, score_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        # Draw stance type
        stance_text = f"Stance: {metrics.stance_type}"
        cv2.putText(
            output, stance_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        return output


def test_balance_analyzer():
    """Test balance analyzer with synthetic keypoints."""
    print("Testing Balance Analyzer...")

    # Create synthetic keypoints (parallel stance)
    from .pose_estimator import PoseKeypoints

    keypoints = PoseKeypoints()
    keypoints.keypoints_2d = {
        'nose': (0.5, 0.2),
        'left_shoulder': (0.45, 0.35),
        'right_shoulder': (0.55, 0.35),
        'left_hip': (0.45, 0.50),
        'right_hip': (0.55, 0.50),
        'left_knee': (0.45, 0.70),
        'right_knee': (0.55, 0.70),
        'left_ankle': (0.40, 0.90),
        'right_ankle': (0.60, 0.90),
        'left_foot': (0.38, 0.95),
        'right_foot': (0.62, 0.95),
    }
    keypoints.visibility = {k: 1.0 for k in keypoints.keypoints_2d.keys()}

    # Analyze
    analyzer = BalanceAnalyzer()
    metrics = analyzer.analyze(keypoints)

    print(f"  Support polygon: {metrics.support_polygon is not None}")
    print(f"  COM: {metrics.center_of_mass}")
    print(f"  COM inside support: {metrics.com_inside_support}")
    print(f"  Balance score: {metrics.balance_score:.1f}")
    print(f"  Stance type: {metrics.stance_type}")
    print(f"  Stance width: {metrics.stance_width:.3f}")

    success = metrics.balance_score > 0 and metrics.com_inside_support
    print(f"  Result: {'PASS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    test_balance_analyzer()
