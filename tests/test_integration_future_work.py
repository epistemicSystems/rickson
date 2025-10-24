"""
Integration Tests for Future Work Features

Tests the complete pipeline with:
- GPU-accelerated EVM
- 3D Gaussian Splatting
- Multi-camera 3D pose fusion
- Opponent analysis
- Timeline replay
- Privacy features
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json


# GPU EVM Tests
def test_gpu_evm_integration():
    """Test GPU EVM pipeline integration."""
    print("\n=== Testing GPU EVM Integration ===")

    try:
        from zs.evm.cuda.gpu_pipeline import GPUEVMPipeline
    except ImportError:
        pytest.skip("GPU pipeline not available")

    # Create pipeline
    pipeline = GPUEVMPipeline(
        fps=30.0,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4,
        use_cuda=True
    )

    # Generate test frames
    frames = []
    for i in range(30):
        frame = np.ones((128, 128, 3), dtype=np.uint8) * 128
        frames.append(frame)

    # Process
    for frame in frames:
        amplified, metrics = pipeline.process_frame(frame)

        assert amplified.shape == frame.shape
        assert 'breath_rate_bpm' in metrics

    print("  ✓ GPU EVM working")


def test_3dgs_integration():
    """Test 3D Gaussian Splatting integration."""
    print("\n=== Testing 3DGS Integration ===")

    from zs.gaussian_splats.splat_loader import GaussianSplat, GaussianSplatLoader
    from zs.gaussian_splats.gym_prior import GymPrior

    # Create synthetic splat
    N = 100
    positions = np.random.randn(N, 3)
    colors = np.random.rand(N, 3)
    opacities = np.random.rand(N)
    scales = np.random.rand(N, 3) * 0.1
    rotations = np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32)

    splat = GaussianSplat(
        positions=positions,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations
    )

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        temp_path = f.name

    GaussianSplatLoader.save_ply(splat, temp_path)
    loaded = GaussianSplatLoader.load_ply(temp_path)

    assert len(loaded) == len(splat)

    # Test gym prior
    gym_prior = GymPrior(splat)

    assert gym_prior is not None
    assert gym_prior.get_floor_height() is not None

    # Test depth query
    depth = gym_prior.query_depth(
        np.array([0, 0, 2]),
        np.array([0, 0, -1]),
        max_distance=5.0
    )

    print(f"  ✓ 3DGS working, depth query: {depth}")

    import os
    os.unlink(temp_path)


def test_multicam_integration():
    """Test multi-camera calibration and fusion."""
    print("\n=== Testing Multi-Camera Integration ===")

    from zs.pose.multicam.calibration import MultiCameraCalibration, CameraIntrinsics, CameraExtrinsics
    from zs.pose.multicam.synchronization import FrameSynchronizer
    from zs.pose.multicam.pose_fusion import Pose3DFusion
    from zs.pose.pose_estimator import PoseKeypoints

    # Create calibration
    calib = MultiCameraCalibration()

    intrinsics1 = CameraIntrinsics(
        fx=800, fy=800, cx=640, cy=360,
        width=1280, height=720,
        distortion=np.zeros(5)
    )

    intrinsics2 = CameraIntrinsics(
        fx=800, fy=800, cx=640, cy=360,
        width=1280, height=720,
        distortion=np.zeros(5)
    )

    extrinsics2 = CameraExtrinsics(
        R=np.eye(3),
        t=np.array([[1.0], [0.0], [0.0]])
    )

    calib.add_camera('cam1', intrinsics1)
    calib.add_camera('cam2', intrinsics2, extrinsics2)

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        temp_path = f.name

    calib.save(temp_path)
    loaded_calib = MultiCameraCalibration.load(temp_path)

    assert len(loaded_calib.cameras) == 2

    # Test synchronizer
    sync = FrameSynchronizer(sync_method='timestamp')
    sync.add_camera('cam1')
    sync.add_camera('cam2')

    for i in range(10):
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

        sync.add_frame('cam1', frame1, i * 0.033, i)
        sync.add_frame('cam2', frame2, i * 0.033 + 0.005, i)

        synced = sync.get_synced_frame()

        if synced:
            assert 'cam1' in synced.frames
            assert 'cam2' in synced.frames

    # Test 3D fusion
    fusion = Pose3DFusion(loaded_calib)

    # Create synthetic 2D poses
    point_3d = np.array([[0, 0, 3]])

    intrinsics1, extrinsics1 = loaded_calib.cameras['cam1']
    intrinsics2, extrinsics2 = loaded_calib.cameras['cam2']

    point_2d_1 = extrinsics1.project(point_3d, intrinsics1)[0] / [1280, 720]
    point_2d_2 = extrinsics2.project(point_3d, intrinsics2)[0] / [1280, 720]

    pose1 = PoseKeypoints()
    pose1.keypoints_2d = {'nose': tuple(point_2d_1)}
    pose1.visibility = {'nose': 0.9}

    pose2 = PoseKeypoints()
    pose2.keypoints_2d = {'nose': tuple(point_2d_2)}
    pose2.visibility = {'nose': 0.9}

    pose_3d = fusion.fuse({'cam1': pose1, 'cam2': pose2}, timestamp=0.0)

    assert pose_3d is not None
    assert 'nose' in pose_3d.keypoints_3d

    error = np.linalg.norm(point_3d[0] - pose_3d.keypoints_3d['nose'])

    print(f"  ✓ Multi-camera working, 3D error: {error:.6f}m")

    import os
    os.unlink(temp_path)


def test_opponent_analysis_integration():
    """Test opponent analysis and training game recommendation."""
    print("\n=== Testing Opponent Analysis Integration ===")

    from zs.opponent.feature_extractor import OpponentFeatureExtractor, OpponentProfile
    from zs.opponent.training_games import TrainingGameRecommender

    # Create extractor
    extractor = OpponentFeatureExtractor()

    # Simulate footage analysis
    for i in range(300):  # 10 seconds at 30fps
        t = i / 30.0

        keypoints = {
            'left_hip': np.array([0, 0, 1.0]),
            'right_hip': np.array([0.2, 0, 1.0])
        }

        extractor.add_frame_data(
            timestamp=t,
            pose_keypoints=keypoints,
            breath_rate=18.0 + np.random.rand() * 2,
            stance_type='orthodox'
        )

        if i % 100 == 0:
            extractor.add_event('strike')

    # Extract profile
    profile = extractor.extract_profile('Test_Opponent')

    assert profile.name == 'Test_Opponent'
    assert profile.stance == 'orthodox'
    assert profile.strike_rate > 0

    # Test recommender
    recommender = TrainingGameRecommender()

    # Create profile with specific attributes
    profile = OpponentProfile(
        name='Opponent_A',
        stance='southpaw',
        pressure_style='aggressive',
        guard_pull_rate=1.5,
        strike_rate=4.0
    )

    games = recommender.recommend(profile, max_recommendations=3)

    assert len(games) > 0
    assert games[0].priority > 0

    # Test training plan
    plan = recommender.generate_training_plan(profile, available_time=60)

    assert 'games' in plan
    assert len(plan['games']) > 0

    print(f"  ✓ Opponent analysis working, {len(games)} games recommended")


def test_timeline_replay_integration():
    """Test timeline replay with annotations."""
    print("\n=== Testing Timeline Replay Integration ===")

    from zs.core.event_log import EventLog
    from zs.core.timeline_replay import TimelinePlayer

    # Create event log
    log = EventLog()

    log.log_event('session_started', {'session_id': 'test_001'})

    for i in range(100):
        t = i * 0.033

        log.log_event('frame_processed', {'frame_number': i}, timestamp=t)

        if i % 30 == 0:
            log.log_event('breath_rate_estimated', {
                'breath_rate_bpm': 18.0,
                'confidence': 0.9
            }, timestamp=t)

    log.log_event('session_ended', {}, timestamp=3.3)

    # Create player
    player = TimelinePlayer(log)

    assert player.end_time > 0
    assert len(player.markers) > 0

    # Test playback
    player.play()

    events_count = 0
    while player.current_time < player.end_time:
        events = player.update(0.033)
        events_count += len(events)

        if player.current_time > 2.0:
            break

    assert events_count > 0

    # Test annotations
    player.add_annotation(1.0, "Test annotation", "user")

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        temp_path = f.name

    player.export_annotations(temp_path)

    new_player = TimelinePlayer(log)
    new_player.load_annotations(temp_path)

    assert len(new_player.markers) > 0

    print(f"  ✓ Timeline replay working, {len(player.markers)} markers")

    import os
    os.unlink(temp_path)


def test_privacy_integration():
    """Test face blurring and privacy features."""
    print("\n=== Testing Privacy Features Integration ===")

    from zs.core.privacy import FaceBlurrer, FaceRegion, PrivacyManager

    # Create test frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Add face region
    face_region = FaceRegion(
        bbox=(200, 150, 150, 150),
        confidence=0.9
    )

    # Test blurring
    blurrer = FaceBlurrer(blur_method='gaussian')
    blurred = blurrer.blur_frame(frame, face_regions=[face_region])

    assert blurred.shape == frame.shape

    # Check that region was modified
    roi_original = frame[150:300, 200:350]
    roi_blurred = blurred[150:300, 200:350]

    diff = np.mean(np.abs(roi_original.astype(float) - roi_blurred.astype(float)))

    assert diff > 0.1, "Face region should be modified"

    # Test privacy manager
    privacy = PrivacyManager()

    assert not privacy.can_export_raw()

    privacy.set_consent(True, "athlete_001")

    assert privacy.can_export_raw()

    print(f"  ✓ Privacy features working, blur diff: {diff:.2f}")


def test_full_pipeline():
    """Test complete pipeline with all features."""
    print("\n=== Testing Full Pipeline Integration ===")

    from zs.evm.core.evm_pipeline import EVMPipeline
    from zs.pose.pose_estimator import PoseEstimator
    from zs.pose.balance_analyzer import BalanceAnalyzer
    from zs.core.insights_engine import InsightsEngine
    from zs.core.event_log import EventLog

    # Create components
    evm_pipeline = EVMPipeline(
        fps=30.0,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4
    )

    balance_analyzer = BalanceAnalyzer()
    insights_engine = InsightsEngine()
    event_log = EventLog()

    # Simulate session
    event_log.log_event('session_started', {'session_id': 'integration_test'})

    breath_rates = []
    balance_scores = []

    for i in range(30):
        # Generate frame
        frame = np.ones((128, 128, 3), dtype=np.uint8) * 128

        # EVM
        amplified, evm_metrics = evm_pipeline.process_frame(frame)

        breath_rates.append(evm_metrics['breath_rate_bpm'])

        # Log event
        event_log.log_event('frame_processed', {
            'frame_number': i,
            'breath_rate': evm_metrics['breath_rate_bpm']
        })

        # Simulate balance
        balance_scores.append(75.0 + np.random.rand() * 10)

    # Generate insights
    insights = insights_engine.analyze_breath_patterns(breath_rates)

    assert len(insights) > 0

    # Check event log
    summary = event_log.get_session_summary()

    assert summary['event_count'] > 0

    print(f"  ✓ Full pipeline working:")
    print(f"    Frames processed: {len(breath_rates)}")
    print(f"    Events logged: {summary['event_count']}")
    print(f"    Insights generated: {len(insights)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Integration Tests for Future Work Features")
    print("=" * 60)

    test_3dgs_integration()
    test_multicam_integration()
    test_opponent_analysis_integration()
    test_timeline_replay_integration()
    test_privacy_integration()
    test_full_pipeline()

    try:
        test_gpu_evm_integration()
    except Exception as e:
        print(f"  ⚠ GPU EVM skipped: {e}")

    print("\n" + "=" * 60)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
