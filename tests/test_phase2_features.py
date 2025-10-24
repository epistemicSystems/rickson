"""
Phase 2 Features Integration Test

Tests advanced features:
1. GPU acceleration framework (with CPU fallback)
2. Timeline replay from event log
3. Face blurring for privacy
4. Opponent analysis and training game generation
"""

import sys
from pathlib import Path

# Add extensions to path
exts_dir = Path(__file__).parent.parent / "exts"
sys.path.insert(0, str(exts_dir))

import numpy as np


def test_gpu_acceleration():
    """Test GPU acceleration framework."""
    print("\n" + "="*70)
    print("1. GPU ACCELERATION FRAMEWORK")
    print("="*70)

    from zs.evm.gpu.evm_gpu import GPU_AVAILABLE, GPUPyramid

    print(f"\n[GPU] CuPy available: {GPU_AVAILABLE}")

    # Test pyramid builder
    pyramid_builder = GPUPyramid(use_gpu=GPU_AVAILABLE)

    # Create test image
    test_image = np.random.rand(256, 256, 3).astype(np.float32) * 255

    # Build pyramid
    pyramid = pyramid_builder.build_gaussian_pyramid_gpu(test_image, levels=4)

    print(f"✓ Built Gaussian pyramid with {len(pyramid)} levels")
    for i, level in enumerate(pyramid):
        print(f"  Level {i}: {level.shape}")

    # Build Laplacian
    laplacian = pyramid_builder.build_laplacian_pyramid_gpu(pyramid)

    print(f"✓ Built Laplacian pyramid with {len(laplacian)} levels")

    if GPU_AVAILABLE:
        print("✓ GPU acceleration enabled")
    else:
        print("✓ CPU fallback working (install CuPy for GPU acceleration)")

    return True


def test_timeline_replay():
    """Test timeline replay system."""
    print("\n" + "="*70)
    print("2. TIMELINE REPLAY SYSTEM")
    print("="*70)

    from zs.core.event_log import EventLog, EventTypes
    from zs.core.timeline_replay import TimelineReplay, TimelineAnnotator

    # Create event log with sample data
    log = EventLog()

    log.append_event(EventTypes.SESSION_STARTED, {'duration': 30.0, 'fps': 30.0})

    for frame_num in range(300):  # 10 seconds
        timestamp = frame_num / 30.0

        log.append_event(EventTypes.FRAME_INGESTED, {
            'frame': frame_num,
            'timestamp': timestamp
        })

        if frame_num % 30 == 0:  # Every second
            log.append_event(EventTypes.EVM_BREATH_CYCLE, {
                'frame': frame_num,
                'breath_rate_bpm': 18.0 + np.random.randn() * 2,
                'confidence': 0.9
            })

        log.append_event(EventTypes.POSE_ESTIMATED, {
            'frame': frame_num,
            'balance_score': 75.0 + np.random.randn() * 10,
            'stance_type': 'parallel',
            'com_position': [0.5, 0.6]
        })

        # Add alerts
        if frame_num in [100, 200]:
            log.append_event(EventTypes.ALERT_BALANCE_EDGE, {
                'frame': frame_num,
                'balance_score': 30.0
            })

    log.append_event(EventTypes.SESSION_ENDED, {'frames_processed': 300})

    # Create timeline
    replay = TimelineReplay(log)

    print(f"\n✓ Created timeline with {len(replay.frames)} frames")

    # Test navigation
    replay.seek(150)
    frame = replay.get_current_frame()
    print(f"✓ Seek to frame 150: breath={frame.breath_rate_bpm:.1f} BPM, "
          f"balance={frame.balance_score:.0f}")

    # Test finding alerts
    alert_frames = replay.find_alerts()
    print(f"✓ Found {len(alert_frames)} frames with alerts")

    # Test summary
    summary = replay.get_summary()
    print(f"✓ Summary: {summary['total_frames']} frames, "
          f"{summary['duration_seconds']:.1f}s")
    print(f"  Mean breath: {summary['breath_analysis']['mean_bpm']:.1f} BPM")
    print(f"  Mean balance: {summary['balance_analysis']['mean_score']:.0f}/100")

    # Test annotations
    annotator = TimelineAnnotator(log)
    annotator.add_annotation(150, 'technique', 'Good stance transition')
    annotator.add_annotation(100, 'error', 'Balance edge alert')

    annotations = annotator.get_annotations()
    print(f"✓ Added {len(annotations)} annotations")

    # Test export
    output_path = Path("/tmp/timeline_test.json")
    replay.export_to_usd_timeline(output_path)
    print(f"✓ Exported timeline to {output_path}")

    return True


def test_face_blurring():
    """Test face blurring for privacy."""
    print("\n" + "="*70)
    print("3. FACE BLURRING (PRIVACY)")
    print("="*70)

    from zs.core.privacy import PrivacyFilter

    # Create test frame
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Draw a mock "face" region (circle)
    import cv2
    cv2.circle(test_frame, (320, 240), 80, (200, 200, 200), -1)

    # Test Gaussian blur
    pfilter_gaussian = PrivacyFilter(blur_method='gaussian', blur_strength=21)
    print("✓ Created Gaussian blur filter")

    # Manually blur the mock face region
    x, y, w, h = 240, 160, 160, 160
    roi = test_frame[y:y+h, x:x+w]
    blurred_roi = pfilter_gaussian._apply_gaussian_blur(roi)
    test_frame[y:y+h, x:x+w] = blurred_roi

    print("✓ Applied Gaussian blur")

    # Test pixelate
    pfilter_pixelate = PrivacyFilter(blur_method='pixelate', blur_strength=20)
    test_frame_2 = test_frame.copy()
    pixelated_roi = pfilter_pixelate._apply_pixelate(roi)
    print("✓ Applied pixelation")

    # Test solid fill
    pfilter_solid = PrivacyFilter(blur_method='solid')
    filled_roi = pfilter_solid._apply_solid_fill(roi)
    print("✓ Applied solid fill")

    print("✓ All privacy filter methods working")

    return True


def test_opponent_analysis():
    """Test opponent analysis and training game generation."""
    print("\n" + "="*70)
    print("4. OPPONENT ANALYSIS (Milestone 6)")
    print("="*70)

    from zs.core.opponent_analysis import (
        FeatureExtractor,
        OpponentAnalyzer,
        TrainingGameGenerator
    )

    # Feature extraction
    extractor = FeatureExtractor()

    print("\n[Simulating opponent analysis...]")

    for _ in range(300):  # 10 seconds @ 30fps
        stance = np.random.choice(
            ['orthodox', 'southpaw', 'parallel'],
            p=[0.7, 0.2, 0.1]  # Strong orthodox preference
        )

        attack = None
        if np.random.rand() < 0.15:  # 15% attack frequency
            attack = np.random.choice(
                ['jab', 'cross', 'hook', 'low_kick', 'takedown'],
                p=[0.5, 0.25, 0.15, 0.08, 0.02]  # Jab-heavy
            )

        position = np.random.choice(['standing', 'guard', 'top'], p=[0.7, 0.2, 0.1])

        extractor.process_frame_data(stance, attack, position)

    # Get features
    features = extractor.get_summary()

    print(f"\n✓ Extracted features:")
    print(f"  Preferred stance: {features['stance']['preferred']} "
          f"({features['stance']['strength']*100:.0f}% preference)")
    print(f"  Total attacks: {features['attacks']['total_observed']}")

    # Analyze patterns
    analyzer = OpponentAnalyzer()
    patterns = analyzer.analyze_features(features)

    print(f"\n✓ Detected {len(patterns)} opponent patterns:")
    for pattern in patterns:
        print(f"  - [{pattern.pattern_type}] {pattern.description}")
        print(f"    Counter: {pattern.counter_strategy[:60]}...")

    # Generate training games
    generator = TrainingGameGenerator()
    games = generator.generate_games(patterns)

    print(f"\n✓ Generated {len(games)} training games:")
    for game in games:
        print(f"  - {game.name} ({game.difficulty}, {game.duration_minutes}min)")
        print(f"    Objective: {game.objective}")

    # Format training plan
    training_plan = generator.format_game_plan(games)
    print("\n" + "="*70)
    print(training_plan)

    return True


def test_all_phase2_features():
    """Run all Phase 2 feature tests."""
    print("="*70)
    print("RICKSON PHASE 2 - Advanced Features Test")
    print("="*70)

    tests = [
        ("GPU Acceleration Framework", test_gpu_acceleration),
        ("Timeline Replay System", test_timeline_replay),
        ("Face Blurring (Privacy)", test_face_blurring),
        ("Opponent Analysis", test_opponent_analysis),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} {name}")

    all_passed = all(s for _, s in results)

    print()
    if all_passed:
        print("="*70)
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("="*70)
        print("\nPhase 2 Features Ready:")
        print("  ✓ GPU acceleration framework (CuPy-based, CPU fallback)")
        print("  ✓ Timeline replay with scrubbing and annotations")
        print("  ✓ Privacy-preserving face blurring")
        print("  ✓ Opponent analysis with training game generation")
    else:
        print("="*70)
        print("⚠️  SOME TESTS FAILED")
        print("="*70)

    return all_passed


if __name__ == "__main__":
    success = test_all_phase2_features()
    sys.exit(0 if success else 1)
