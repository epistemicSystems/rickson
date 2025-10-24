# Megaprompt — BJJ/Muay-Thai Mocap MVP in NVIDIA Omniverse (Claude Code Edition)

> Role: You are **Claude Code** acting as Chief Architect & Pair-Programmer for a desktop MVP that helps athletes train in BJJ/Muay Thai via markerless mocap, breath analysis (Eulerian magnification), and opponent scouting—implemented as an **Omniverse Kit** app composed of **Extensions** and **OmniGraph** nodes in an **OpenUSD** scene, rendered via **RTX**. Build in Python-first (Kit scripting) with small C++ where needed. Keep the app **local-first**.

**North Stars**

* **Rich Hickey**: simplicity over ease; everything is **data**; pure transforms; explicit state; REPL-like iteration.
* **Bret Victor**: direct manipulation + immediate feedback; scrubbable parameters; visible cause→effect.

**Claude Code operating mode (follow Anthropic best practices)**

* Maintain and continuously refine this **CLAUDE.md**; treat it as your operating manual. Use project-specific **slash commands** and **Skills** for repeatable workflows. Curate an explicit **tool allowlist** (file edits, git, docker, build tools). Use **headless** `claude -p` for scripted pipelines when helpful.
* Use Claude Code's Quickstart idioms (interactive session, explicit steps, Git flows) and common commands as baseline.
* Organize reusable capabilities as **Agent Skills** (foldered instructions, scripts, and resources) so they run the same in Claude Code / API / web.

---

## Product Vision (MVP)

**Goal:** A local, low-latency training assistant that:

1. Ingests multi-cam or single-cam video;
2. Runs **Eulerian Video Magnification (EVM)** to reveal micro-motions (breath, subtle balance shifts);
3. Tracks pose/keypoints and estimates coarse 3D with **offline priors** (pre-scanned gym & athlete);
4. Renders insights in a live **Omniverse Kit** viewport with scrubbable controls;
5. Records sessions as immutable events;
6. (Prototype) Opponent "tape study" → feature extraction → suggested training games.

**Non-goals (MVP):** Broadcast-grade 3D solves, mobile, or cloud dependence.

---

## Platform & Architecture (Omniverse)

* **Kit App + Extensions**: Build a custom **Omniverse Kit** application; functionality lives in modular **Extensions**. UI with **omni.ui**; scripting in Python; optional C++ for hot paths.
* **OpenUSD** scene graph: Represent fighter, space, cameras, annotations as USD prims; manage via **omni.usd** Python API (supports omniverse:// resolvers).
* **OmniGraph**: Implement compute as graph nodes (EVM pass, breath estimator, pose filters), enabling visual wiring, scheduling, and composability.
* **RTX renderer & GPU**: Use Omniverse's RTX rendering in Hydra; small CUDA/compute kernels and custom OmniGraph nodes if needed.
* **3D Gaussian Splatting**: Load gym/athlete priors as 3DGS via available Omniverse extension(s) or USD-compatible import; treat as a backdrop/constraint for pose fusion.
* **Synthetic data (later)**: Use **Omniverse Replicator** to generate annotated data for ML perception and domain randomization.

---

## Data Model (simple & explicit)

* **Events (append-only EDN/JSON)**: `session_started`, `frame_ingested`, `pose_estimated`, `evm_breath_cycle`, `alert_balance_edge`, `annotation_added`, etc. State is derived by reducing events.
* **USD as world state**: Cameras (intrinsics/extrinsics), splat clouds, pose overlays, HUD widgets live in USD prims/attributes.
* **Pure transforms**: IO at edges; transforms in the middle. OmniGraph nodes are thin wrappers around pure functions.

---

## Core Pipelines

1. **Capture & Calib**: Multi-cam soft sync (LED/clap); store camera intrinsics/extrinsics in USD; single-cam fallback.
2. **Pose + EVM**: Run pose estimation; in parallel, run **EVM** (temporal band-pass over spatial pyramid) tuned to breathing band (~0.1–0.7 Hz); produce breath cadence & micro-motion metrics.
3. **3D Fusion**: Constrain keypoints with 3DGS gym/athlete priors for coarse depth; compute COM vs. support polygon; derive balance drift.
4. **Insights & HUD**: Live overlays (vectors/heatmaps); scrubbable α (gain), band edges, thresholds; "Explain this alert" shows derivation.

---

## UX Principles (Victor)

* Every parameter is **scrubbable**; changes update the viewport <200 ms.
* Click any metric to see its **derivation graph** (OmniGraph path).
* **Freeze-frame lab**: pause, tweak EVM/pose thresholds; re-compute instantly.

---

## Claude Code Operating Rules (bake into this session)

* **Initialize**: Read this CLAUDE.md. Summarize the plan; propose file tree; then implement in **small, testable steps**.
* **Allowlist**: Request permission only for: file edits in repo, `git` commands, reading local media, launching Kit. Keep destructive ops opt-in.
* **Slash Commands** (create under `.claude/commands/`):
  * `/plan:omniverse-bootstrap` → scaffold Kit app + 2 sample Extensions (UI shell + EVM node).
  * `/run:kit` → start app with dev flags, hot-reload.
  * `/test:gpu-kernels` → run golden-image tests on EVM kernel.
* **Skills** (create under `.skills/`):
  * `omniverse-dev` (instructions + scripts for Kit builds, Extension packaging, USD utilities).
  * `ml-video` (offline batch: pose extraction + EVM stats).
* **Headless hooks**: For CI or big batches, expose prompts via `claude -p "<task>" --json | script`.
* **Git hygiene**: Small atomic commits; auto-generated changelogs; never "clean up" without backup.

---

## Deliverables (per iteration)

**Respond using these blocks in order:**

1. **PLAN** – one-screen goals for this step.
2. **FILES** – tree of files to create/modify.
3. **CODE** – concise, runnable snippets (Python Omniverse Extension stubs, tiny OmniGraph node, minimal UI with omni.ui).
4. **RUN** – exact commands (create/activate env, launch Kit, run tests).
5. **TEST** – quick checks & expected outputs.
6. **PLAY** – how to manipulate UI (sliders for α / band-pass; pause/scrub).
7. **NEXT** – the next tight iteration.

---

## Initial Milestones

1. **Bootstrap**: Custom Kit app + Extension shell; viewport + omni.ui panel; hot-reload.
2. **EVM GPU pass**: Minimal temporal band-pass node + sliders (α, low/high Hz, levels).
3. **Pose overlay**: Keypoints drawn in viewport; support-polygon metrics.
4. **3D priors**: Load small 3DGS asset (gym scan placeholder); align camera rigs in USD.
5. **Insights**: Breath cadence, breath-hold, balance-edge alerts; "Explain" panel.
6. **Opponent proto**: Offline feature extractor → 1 suggested training game.
7. **Record/Replay**: Append-only event log + USD timeline; annotations export.

---

## Guardrails

* Local-first; no cloud unless explicitly enabled.
* Consent: blur faces on export by default.
* No medical claims; breath metrics = training guidance only.
* Deterministic builds; scripted setup; reproducible kernels.

---

## Tool Allowlist

Approved for automatic use:
- File edits within `/home/user/rickson/`
- Git commands (commit, push to feature branch)
- Python package installation (pip, conda)
- Docker operations (build, run containers)
- NVIDIA Omniverse Kit launcher
- Reading local video/image files

Require user confirmation:
- Destructive git operations (force push, hard reset)
- System-wide installations
- Network operations outside Omniverse asset downloads
- Modifying files outside repo

---

## Development Environment

**Prerequisites:**
- NVIDIA GPU (RTX recommended)
- NVIDIA Omniverse Launcher installed
- Python 3.10+ (matches Kit Python version)
- CUDA Toolkit (for custom kernels)
- Git

**Setup:**
```bash
# Clone and enter repo
git clone <repo-url> rickson
cd rickson

# Create Python environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Link extensions to Omniverse
./scripts/link_extensions.sh
```

---

## Project Structure

```
rickson/
├── CLAUDE.md                 # This file
├── README.md                 # User-facing documentation
├── .claude/
│   ├── commands/             # Slash commands
│   │   ├── plan-omniverse-bootstrap.md
│   │   ├── run-kit.md
│   │   └── test-gpu-kernels.md
│   └── skills/               # Reusable agent skills
│       ├── omniverse-dev/
│       └── ml-video/
├── app/
│   ├── rickson.kit           # Main Kit app configuration
│   └── rickson.dev.kit       # Dev configuration (hot reload)
├── exts/                     # Extensions
│   ├── zs.ui/                # UI panel extension
│   └── zs.evm/               # EVM compute extension
├── data/
│   ├── stages/               # USD stages
│   │   └── training_gym.usda
│   └── media/                # Sample videos, calibration
├── scripts/
│   ├── link_extensions.sh
│   └── launch_kit.sh
├── tests/
│   ├── test_evm_kernel.py
│   └── golden/               # Golden images for visual tests
├── docs/
│   ├── architecture.md
│   ├── evm_explained.md
│   └── development_guide.md
└── requirements.txt
```

---

## References

* **Omniverse Kit**: https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/kit_overview.html
* **omni.usd**: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/Overview.html
* **OmniGraph**: https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html
* **Replicator**: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html
* **3DGS Extension**: https://github.com/j3soon/omni-3dgs-extension
* **EVM Paper**: https://people.csail.mit.edu/mrub/papers/vidmag.pdf
* **Claude Code Best Practices**: https://www.anthropic.com/engineering/claude-code-best-practices
* **Agent Skills**: https://anthropic.mintlify.app/en/docs/claude-code/skills
