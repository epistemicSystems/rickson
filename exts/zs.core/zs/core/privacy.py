"""
Privacy-Preserving Export

Face blurring and anonymization for video export.
Ensures consent and privacy compliance.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FaceRegion:
    """
    Detected face region.

    Attributes:
        bbox: Bounding box (x, y, w, h)
        confidence: Detection confidence
        landmarks: Optional facial landmarks
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray] = None


class FaceBlurrer:
    """
    Face detection and blurring for privacy.

    Supports:
    - Automatic face detection
    - Gaussian blur
    - Pixelation
    - Black boxes
    """

    def __init__(
        self,
        blur_method: str = 'gaussian',
        blur_strength: int = 51,
        min_face_size: int = 30
    ):
        """
        Initialize face blurrer.

        Args:
            blur_method: 'gaussian', 'pixelate', or 'blackbox'
            blur_strength: Blur kernel size (odd number)
            min_face_size: Minimum face size to detect (pixels)
        """
        self.blur_method = blur_method
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.min_face_size = min_face_size

        # Load face detector (use Haar cascade as fallback)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            self.face_cascade = None
            print("[Privacy] Warning: Face cascade not available")

    def detect_faces(self, frame: np.ndarray) -> List[FaceRegion]:
        """
        Detect faces in frame.

        Args:
            frame: Input frame (H, W, 3)

        Returns:
            List of FaceRegion
        """
        if self.face_cascade is None:
            # Fall back to pose-based detection
            return self._detect_faces_from_pose(frame)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )

        face_regions = []

        for (x, y, w, h) in faces:
            face_regions.append(FaceRegion(
                bbox=(x, y, w, h),
                confidence=1.0
            ))

        return face_regions

    def _detect_faces_from_pose(self, frame: np.ndarray) -> List[FaceRegion]:
        """
        Estimate face region from pose keypoints (fallback).

        Args:
            frame: Input frame

        Returns:
            List of estimated face regions
        """
        # Would use pose estimator here
        # For now, return empty
        return []

    def blur_frame(
        self,
        frame: np.ndarray,
        face_regions: Optional[List[FaceRegion]] = None
    ) -> np.ndarray:
        """
        Blur faces in frame.

        Args:
            frame: Input frame
            face_regions: Optional pre-detected faces (will detect if None)

        Returns:
            Blurred frame
        """
        if face_regions is None:
            face_regions = self.detect_faces(frame)

        if not face_regions:
            return frame

        output = frame.copy()

        for face in face_regions:
            x, y, w, h = face.bbox

            # Expand region slightly
            padding = int(h * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)

            # Extract face region
            face_roi = output[y:y+h, x:x+w]

            if face_roi.size == 0:
                continue

            # Apply blur method
            if self.blur_method == 'gaussian':
                blurred = cv2.GaussianBlur(face_roi, (self.blur_strength, self.blur_strength), 0)

            elif self.blur_method == 'pixelate':
                # Pixelate by downsampling and upsampling
                pixel_size = 10
                small_h = max(1, h // pixel_size)
                small_w = max(1, w // pixel_size)

                small = cv2.resize(face_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

            elif self.blur_method == 'blackbox':
                # Black box
                blurred = np.zeros_like(face_roi)

            else:
                blurred = face_roi

            # Replace region
            output[y:y+h, x:x+w] = blurred

        return output

    def blur_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[callable] = None
    ):
        """
        Blur faces in video file.

        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Optional callback(frame_num, total_frames)
        """
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Blur frame
            blurred = self.blur_frame(frame)

            # Write
            out.write(blurred)

            frame_num += 1

            if progress_callback:
                progress_callback(frame_num, total_frames)

        cap.release()
        out.release()


class PrivacyManager:
    """
    Manage privacy settings and consent for recordings.
    """

    def __init__(self):
        """Initialize privacy manager."""
        self.default_blur_on_export = True
        self.consent_obtained = False
        self.participant_ids: List[str] = []

    def set_consent(self, obtained: bool, participant_id: Optional[str] = None):
        """
        Record consent.

        Args:
            obtained: Whether consent obtained
            participant_id: Optional participant identifier
        """
        self.consent_obtained = obtained

        if participant_id and obtained:
            self.participant_ids.append(participant_id)

    def can_export_raw(self) -> bool:
        """Check if raw (unblurred) export is allowed."""
        return self.consent_obtained

    def export_video(
        self,
        input_path: str,
        output_path: str,
        blur_faces: Optional[bool] = None,
        blur_method: str = 'gaussian'
    ):
        """
        Export video with privacy settings.

        Args:
            input_path: Input video
            output_path: Output video
            blur_faces: Whether to blur (default: use setting)
            blur_method: Blur method
        """
        if blur_faces is None:
            blur_faces = self.default_blur_on_export or not self.consent_obtained

        if blur_faces:
            # Blur faces
            blurrer = FaceBlurrer(blur_method=blur_method)
            blurrer.blur_video(input_path, output_path)
        else:
            # Copy raw
            import shutil
            shutil.copy(input_path, output_path)


def test_face_blurrer():
    """Test face blurring."""
    print("Testing Face Blurrer...")

    # Create synthetic frame with "face" region
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Draw a simple face region (circle)
    center = (320, 240)
    radius = 80

    cv2.circle(frame, center, radius, (200, 180, 160), -1)  # Skin tone
    cv2.circle(frame, (300, 220), 10, (50, 50, 50), -1)  # Left eye
    cv2.circle(frame, (340, 220), 10, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(frame, (320, 260), (30, 15), 0, 0, 180, (150, 100, 100), -1)  # Mouth

    # Create blurrer
    blurrer = FaceBlurrer(blur_method='gaussian', blur_strength=51)

    # Manually specify face region (since Haar cascade might not detect synthetic face)
    face_region = FaceRegion(
        bbox=(center[0] - radius, center[1] - radius, radius * 2, radius * 2),
        confidence=1.0
    )

    # Blur
    blurred = blurrer.blur_frame(frame, face_regions=[face_region])

    print(f"Original frame shape: {frame.shape}")
    print(f"Blurred frame shape: {blurred.shape}")

    # Check that region was modified
    roi_original = frame[
        center[1] - radius:center[1] + radius,
        center[0] - radius:center[0] + radius
    ]

    roi_blurred = blurred[
        center[1] - radius:center[1] + radius,
        center[0] - radius:center[0] + radius
    ]

    diff = np.mean(np.abs(roi_original.astype(float) - roi_blurred.astype(float)))

    print(f"Region difference: {diff:.2f}")

    assert diff > 1.0, "Face region was not blurred"

    # Test privacy manager
    privacy = PrivacyManager()

    print(f"\nPrivacy manager:")
    print(f"  Can export raw: {privacy.can_export_raw()}")

    privacy.set_consent(True, "athlete_001")

    print(f"  After consent: {privacy.can_export_raw()}")

    print("✓ PASS")


if __name__ == "__main__":
    test_face_blurrer()
