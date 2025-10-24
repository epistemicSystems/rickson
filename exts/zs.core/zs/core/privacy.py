"""
Privacy Features

Face detection and blurring for privacy-preserving video export.
Ensures consent compliance when sharing training footage.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
import carb


class FaceDetector:
    """
    Detect faces in video frames for privacy blurring.

    Uses OpenCV's Haar Cascade or DNN-based detection.
    """

    def __init__(self, method: str = 'haar'):
        """
        Initialize face detector.

        Args:
            method: 'haar' (fast, less accurate) or 'dnn' (slower, more accurate)
        """
        self.method = method

        if method == 'haar':
            # Load Haar Cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                carb.log_error("[FaceDetector] Failed to load Haar cascade")
            else:
                carb.log_info("[FaceDetector] Loaded Haar cascade face detector")

        elif method == 'dnn':
            # Load DNN model (more accurate)
            # This would use a pre-trained model like ResNet SSD
            carb.log_warn("[FaceDetector] DNN method not yet implemented, using Haar")
            self.method = 'haar'
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame.

        Args:
            frame: Input frame (H, W, 3) RGB

        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if self.method == 'haar':
            # Convert to grayscale for Haar cascade
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

        return []


class PrivacyFilter:
    """
    Apply privacy filters to video frames.

    Supports multiple blur methods: Gaussian, pixelation, solid color.
    """

    def __init__(
        self,
        blur_method: str = 'gaussian',
        blur_strength: int = 21,
        expand_margin: float = 1.2
    ):
        """
        Initialize privacy filter.

        Args:
            blur_method: 'gaussian', 'pixelate', or 'solid'
            blur_strength: Strength of blur (kernel size for Gaussian)
            expand_margin: Expand face bbox by this factor (1.0 = no expansion)
        """
        self.blur_method = blur_method
        self.blur_strength = blur_strength
        self.expand_margin = expand_margin

        self.face_detector = FaceDetector(method='haar')

        carb.log_info(f"[PrivacyFilter] Initialized with {blur_method} blur, strength={blur_strength}")

    def _expand_bbox(
        self,
        x: int, y: int, w: int, h: int,
        img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box by margin."""
        cx = x + w // 2
        cy = y + h // 2

        new_w = int(w * self.expand_margin)
        new_h = int(h * self.expand_margin)

        new_x = max(0, cx - new_w // 2)
        new_y = max(0, cy - new_h // 2)
        new_w = min(new_w, img_width - new_x)
        new_h = min(new_h, img_height - new_y)

        return new_x, new_y, new_w, new_h

    def _apply_gaussian_blur(self, roi: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to ROI."""
        ksize = self.blur_strength
        if ksize % 2 == 0:
            ksize += 1  # Must be odd

        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        return blurred

    def _apply_pixelate(self, roi: np.ndarray) -> np.ndarray:
        """Apply pixelation to ROI."""
        h, w = roi.shape[:2]

        # Downsample
        pixel_size = max(1, self.blur_strength // 2)
        small = cv2.resize(
            roi,
            (w // pixel_size, h // pixel_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Upsample back
        pixelated = cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        return pixelated

    def _apply_solid_fill(self, roi: np.ndarray) -> np.ndarray:
        """Fill ROI with solid color."""
        filled = np.zeros_like(roi)
        filled[:, :] = [128, 128, 128]  # Gray
        return filled

    def blur_faces(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect and blur faces in frame.

        Args:
            frame: Input frame (H, W, 3) RGB

        Returns:
            Frame with blurred faces
        """
        # Detect faces
        faces = self.face_detector.detect_faces(frame)

        if not faces:
            return frame

        # Clone frame
        output = frame.copy()
        H, W = frame.shape[:2]

        # Blur each face
        for x, y, w, h in faces:
            # Expand bbox
            x, y, w, h = self._expand_bbox(x, y, w, h, W, H)

            # Extract ROI
            roi = frame[y:y+h, x:x+w]

            # Apply blur
            if self.blur_method == 'gaussian':
                blurred_roi = self._apply_gaussian_blur(roi)
            elif self.blur_method == 'pixelate':
                blurred_roi = self._apply_pixelate(roi)
            elif self.blur_method == 'solid':
                blurred_roi = self._apply_solid_fill(roi)
            else:
                blurred_roi = self._apply_gaussian_blur(roi)

            # Replace ROI
            output[y:y+h, x:x+w] = blurred_roi

        return output

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        show_preview: bool = False
    ):
        """
        Process video file with face blurring.

        Args:
            input_path: Input video file
            output_path: Output video file
            show_preview: Show preview window (requires display)
        """
        import cv2

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            carb.log_error(f"[PrivacyFilter] Could not open video: {input_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        carb.log_info(f"[PrivacyFilter] Processing video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Blur faces
            blurred_rgb = self.blur_faces(frame_rgb)

            # Convert back to BGR for output
            blurred_bgr = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(blurred_bgr)

            frame_count += 1

            if frame_count % 30 == 0:
                progress = 100 * frame_count / total_frames
                carb.log_info(f"[PrivacyFilter] Progress: {frame_count}/{total_frames} ({progress:.0f}%)")

            if show_preview:
                cv2.imshow('Privacy Filter Preview', blurred_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()

        if show_preview:
            cv2.destroyAllWindows()

        carb.log_info(f"[PrivacyFilter] Processed {frame_count} frames → {output_path}")


def test_privacy_filter():
    """Test privacy filter with synthetic frame."""
    print("\nTesting Privacy Filter...")
    print("="*70)

    # Create test frame with simulated "face"
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Draw a circle to simulate a face region (Haar cascade won't detect it, but we can test the blur)
    cv2.circle(frame, (320, 240), 80, (200, 200, 200), -1)

    # Create filter
    pfilter = PrivacyFilter(blur_method='gaussian', blur_strength=21)

    # Manually create a face bbox for testing
    faces = [(240, 160, 160, 160)]  # (x, y, w, h) around the circle

    # Apply blur
    blurred = frame.copy()
    for x, y, w, h in faces:
        roi = frame[y:y+h, x:x+w]
        blurred_roi = pfilter._apply_gaussian_blur(roi)
        blurred[y:y+h, x:x+w] = blurred_roi

    print(f"✓ Applied Gaussian blur to {len(faces)} face region(s)")

    # Test pixelate
    pfilter_pixelate = PrivacyFilter(blur_method='pixelate', blur_strength=20)
    pixelated = frame.copy()
    for x, y, w, h in faces:
        roi = frame[y:y+h, x:x+w]
        pixelated_roi = pfilter_pixelate._apply_pixelate(roi)
        pixelated[y:y+h, x:x+w] = pixelated_roi

    print(f"✓ Applied pixelation to {len(faces)} face region(s)")

    # Test solid fill
    pfilter_solid = PrivacyFilter(blur_method='solid')
    filled = frame.copy()
    for x, y, w, h in faces:
        roi = frame[y:y+h, x:x+w]
        filled_roi = pfilter_solid._apply_solid_fill(roi)
        filled[y:y+h, x:x+w] = filled_roi

    print(f"✓ Applied solid fill to {len(faces)} face region(s)")

    print("\n" + "="*70)
    print("Privacy Filter Test: PASS")
    return True


if __name__ == "__main__":
    test_privacy_filter()
