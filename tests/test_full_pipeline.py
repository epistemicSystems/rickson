"""
Full Pipeline Integration Test

Tests the complete Rickson MVP pipeline:
1. Video input (synthetic)
2. EVM breath analysis
3. Pose estimation
4. Balance analysis
5. Insights generation
6. Event logging
"""

import sys
from pathlib import Path

# Add extensions to path
exts_dir = Path(__file__).parent.parent / "exts"
sys.path.insert(0, str(exts_dir))

import numpy as np
from zs.evm.core.evm_pipeline import EVMPipeline
from zs.evm.video_input import SyntheticVideoSource
from zs.pose.pose_estimator import PoseEstimator, MEDIAPIPE_AVAILABLE
from zs.pose.balance_analyzer import BalanceAnalyzer
from zs.core.event_log import EventLog, EventTypes
from zs.core.insights_engine import InsightsEngine


def test_full_pipeline():
    """Test complete pipeline integration."""
    print("="*70)
    print("RICKSON MVP - Full Pipeline Test")
    print("="*70)

    # Configuration
    FPS = 30.0
    DURATION = 10.0  # seconds
    WIDTH, HEIGHT = 640, 480

    print(f"\nConfiguration:")
    print(f"  Duration: {DURATION}s")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  FPS: {FPS}")

    # Initialize components
    print("\n" + "="*70)
    print("1. INITIALIZING COMPONENTS")
    print("="*70)

    # Video source
    print("\n[Video] Creating synthetic video source...")
    video_source = SyntheticVideoSource(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        duration=DURATION,
        breath_freq=0.3  # 18 BPM
    )
    print(f"  ✓ Synthetic video: {video_source.total_frames} frames")

    # EVM pipeline
    print("\n[EVM] Initializing breath analysis pipeline...")
    evm_pipeline = EVMPipeline(
        fps=FPS,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4,
        buffer_seconds=5.0
    )
    print(f"  ✓ EVM pipeline ready")

    # Pose estimator (if available)
    pose_estimator = None
    balance_analyzer = None
    if MEDIAPIPE_AVAILABLE:
        print("\n[Pose] Initializing pose estimator...")
        try:
            pose_estimator = PoseEstimator(model_complexity=0)  # Lite model
            balance_analyzer = BalanceAnalyzer()
            print(f"  ✓ Pose estimator ready")
        except Exception as e:
            print(f"  ✗ Pose estimator failed: {e}")
            pose_estimator = None
    else:
        print("\n[Pose] MediaPipe not available (skipping pose/balance)")

    # Event log
    print("\n[Events] Initializing event log...")
    event_log = EventLog()
    print(f"  ✓ Event log ready")

    # Insights engine
    print("\n[Insights] Initializing insights engine...")
    insights_engine = InsightsEngine()
    print(f"  ✓ Insights engine ready")

    # Processing loop
    print("\n" + "="*70)
    print("2. PROCESSING VIDEO")
    print("="*70)

    event_log.append_event(EventTypes.SESSION_STARTED, {
        'duration': DURATION,
        'fps': FPS,
        'resolution': (WIDTH, HEIGHT)
    })

    frame_count = 0
    breath_rate_history = []
    balance_score_history = []
    stance_type_history = []

    print(f"\nProcessing {video_source.total_frames} frames...")

    while True:
        frame = video_source.read_frame()
        if frame is None:
            break

        frame_count += 1

        # EVM processing
        amplified_frame, evm_metrics = evm_pipeline.process_frame(frame)

        breath_rate = evm_metrics['breath_rate_bpm']
        if breath_rate > 0:
            breath_rate_history.append(breath_rate)

        # Log significant breath measurements
        if frame_count % 30 == 0 and breath_rate > 0:
            event_log.append_event(EventTypes.EVM_BREATH_CYCLE, {
                'frame': frame_count,
                'breath_rate_bpm': breath_rate,
                'confidence': evm_metrics['breath_confidence']
            })

        # Pose estimation (if available)
        if pose_estimator is not None:
            try:
                keypoints = pose_estimator.estimate(frame)

                if keypoints is not None:
                    # Balance analysis
                    balance_metrics = balance_analyzer.analyze(keypoints)

                    balance_score_history.append(balance_metrics.balance_score)
                    stance_type_history.append(balance_metrics.stance_type)

                    # Log pose
                    if frame_count % 30 == 0:
                        event_log.append_event(EventTypes.POSE_ESTIMATED, {
                            'frame': frame_count,
                            'balance_score': balance_metrics.balance_score,
                            'stance_type': balance_metrics.stance_type,
                            'com_inside_support': balance_metrics.com_inside_support
                        })

                    # Check for balance alerts
                    if balance_analyzer.detect_balance_edge_alert():
                        event_log.append_event(EventTypes.ALERT_BALANCE_EDGE, {
                            'frame': frame_count,
                            'balance_score': balance_metrics.balance_score
                        })
            except Exception as e:
                pass  # Skip pose errors

        # Progress
        if frame_count % 60 == 0:
            print(f"  Frame {frame_count}/{video_source.total_frames} "
                  f"({100*frame_count/video_source.total_frames:.0f}%) - "
                  f"Breath: {breath_rate:.1f} BPM")

    event_log.append_event(EventTypes.SESSION_ENDED, {
        'frames_processed': frame_count,
        'duration': DURATION
    })

    print(f"\n✓ Processed {frame_count} frames")

    # Results
    print("\n" + "="*70)
    print("3. RESULTS")
    print("="*70)

    # EVM Results
    print("\n[EVM Breath Analysis]")
    if breath_rate_history:
        print(f"  Measurements: {len(breath_rate_history)}")
        print(f"  Mean breath rate: {np.mean(breath_rate_history):.1f} BPM")
        print(f"  Std dev: {np.std(breath_rate_history):.1f} BPM")
        print(f"  Range: {np.min(breath_rate_history):.1f} - {np.max(breath_rate_history):.1f} BPM")
        print(f"  Expected: ~18 BPM (synthetic data)")

        error = abs(np.mean(breath_rate_history) - 18.0)
        status = "✓ PASS" if error < 3.0 else "✗ FAIL"
        print(f"  Error: {error:.1f} BPM {status}")
    else:
        print(f"  ✗ No breath measurements")

    # Balance Results
    if balance_score_history:
        print("\n[Balance Analysis]")
        print(f"  Measurements: {len(balance_score_history)}")
        print(f"  Mean balance score: {np.mean(balance_score_history):.1f}/100")
        print(f"  Std dev: {np.std(balance_score_history):.1f}")
        print(f"  Range: {np.min(balance_score_history):.1f} - {np.max(balance_score_history):.1f}")

        # Stance distribution
        from collections import Counter
        stance_counts = Counter(stance_type_history)
        print(f"  Stance distribution:")
        for stance, count in stance_counts.most_common():
            pct = 100 * count / len(stance_type_history)
            print(f"    {stance}: {count} ({pct:.0f}%)")

    # Event Log
    print("\n[Event Log]")
    print(f"  Total events: {len(event_log.events)}")
    event_types = {}
    for event in event_log.events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

    for event_type, count in sorted(event_types.items()):
        print(f"    {event_type}: {count}")

    # Insights
    print("\n" + "="*70)
    print("4. TRAINING INSIGHTS")
    print("="*70)

    summary = insights_engine.generate_session_summary(
        breath_rate_history,
        balance_score_history,
        stance_type_history,
        DURATION
    )

    print(f"\nSession Duration: {summary['duration_minutes']:.1f} minutes")

    if summary['insights']:
        print(f"\nGenerated {len(summary['insights'])} insights:\n")
        for i, insight in enumerate(summary['insights'], 1):
            emoji = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'alert': '⚡'
            }.get(insight['severity'], '')

            print(f"{i}. {emoji} {insight['title']} [{insight['category']}]")
            print(f"   {insight['description']}")
            print(f"   → {insight['recommendation']}")
            print(f"   Confidence: {insight['confidence']*100:.0f}%\n")
    else:
        print("\nNo insights generated (need more data)")

    # Summary
    print("\n" + "="*70)
    print("5. TEST SUMMARY")
    print("="*70)

    checks = []

    # Check 1: EVM breath detection
    if breath_rate_history:
        error = abs(np.mean(breath_rate_history) - 18.0)
        checks.append(("EVM breath detection", error < 3.0))
    else:
        checks.append(("EVM breath detection", False))

    # Check 2: Balance analysis
    if balance_score_history:
        checks.append(("Balance analysis", True))
    else:
        checks.append(("Balance analysis", pose_estimator is None))  # OK if pose not available

    # Check 3: Event logging
    checks.append(("Event logging", len(event_log.events) > 0))

    # Check 4: Insights generation
    checks.append(("Insights generation", len(summary['insights']) > 0))

    print()
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} {name}")

    all_passed = all(p for _, p in checks)
    print()
    if all_passed:
        print("="*70)
        print("🎉 ALL TESTS PASSED - MVP READY!")
        print("="*70)
    else:
        print("="*70)
        print("⚠️  SOME TESTS FAILED")
        print("="*70)

    return all_passed


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
