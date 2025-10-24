# Eulerian Video Magnification (EVM) Explained

## What is EVM?

Eulerian Video Magnification (EVM) is a technique to reveal subtle color and motion changes in video that are invisible to the naked eye. It's called "Eulerian" because it works in the spatial domain at each pixel location over time, rather than tracking motion paths (Lagrangian).

**Original paper:** Wu et al., "Eulerian Video Magnification for Revealing Subtle Changes in the World" (SIGGRAPH 2012)
https://people.csail.mit.edu/mrub/papers/vidmag.pdf

## Why EVM for Martial Arts Training?

In BJJ and Muay Thai, subtle cues are critical:

1. **Breathing patterns**: Rhythm, rate, and breath-holds affect performance
2. **Micro-balance shifts**: Small weight transfers before major movements
3. **Tension detection**: Visible in skin color changes (blood flow)
4. **Fatigue signals**: Breathing rate increase, posture micro-changes

EVM makes these invisible cues visible and measurable.

## How EVM Works

### High-Level Pipeline

```
Input Video → Spatial Decomposition → Temporal Filtering → Amplification → Reconstruction
```

### Step-by-Step

#### 1. Spatial Decomposition (Laplacian Pyramid)

Break each frame into multiple spatial frequency bands using a Laplacian pyramid:

```
Original Frame (Level 0) - 1920x1080
    ↓ Gaussian blur + downsample
Level 1 - 960x540
    ↓ Gaussian blur + downsample
Level 2 - 480x270
    ↓ Gaussian blur + downsample
Level 3 - 240x135
    ↓ Gaussian blur + downsample
Level 4 - 120x68
```

**Why?** Different spatial frequencies contain different motion/color information. Breathing affects lower frequencies (larger spatial scales).

#### 2. Temporal Filtering (Band-Pass IIR)

At each level of the pyramid, apply a temporal band-pass filter to each pixel's intensity over time.

**Example:** For breathing at ~0.3 Hz (18 breaths/min):

```
Low cutoff:  0.1 Hz (6 breaths/min)   - filter out slower changes
High cutoff: 0.7 Hz (42 breaths/min)  - filter out faster changes
```

This isolates motion/color changes in the breathing frequency band.

**Implementation:**
- Use IIR (Infinite Impulse Response) filter for real-time processing
- Butterworth band-pass is common choice (flat passband, good rolloff)
- Maintain temporal buffer (e.g., last 10 seconds = 300 frames @ 30fps)

#### 3. Amplification

Multiply the filtered signal by an amplification factor α (alpha):

```
amplified(x, y, t) = filtered(x, y, t) × α
```

**Typical α values:**
- Breath visualization: 10-30
- Pulse detection: 50-100
- Micro-motion: 5-20

**Trade-off:**
- Higher α → more visible magnification
- But also amplifies noise
- Too high → artifacts, temporal flickering

#### 4. Reconstruction

Add the amplified signal back to the original frame:

```
output(x, y, t) = original(x, y, t) + amplified(x, y, t)
```

Then collapse the pyramid back to full resolution.

## Mathematical Details

### Spatial Pyramid

**Gaussian pyramid level i:**
```
G_i = downsample(blur(G_{i-1}))
where G_0 = original frame
```

**Laplacian level i:**
```
L_i = G_i - upsample(G_{i+1})
```

The Laplacian captures the "details" at scale i.

### Temporal Filter

**Band-pass IIR (second-order Butterworth):**

```
b, a = butter(N=2, Wn=[f_low, f_high], btype='band', fs=fps)
filtered[t] = b[0]*input[t] + b[1]*input[t-1] + b[2]*input[t-2]
              - a[1]*filtered[t-1] - a[2]*filtered[t-2]
```

Where:
- `b` = numerator coefficients
- `a` = denominator coefficients
- `f_low`, `f_high` = cutoff frequencies (Hz)
- `fps` = sampling rate (frames per second)

### Amplification

```
amplified_pyramid[level][x,y,t] = filtered_pyramid[level][x,y,t] × α
```

α can be:
- **Uniform**: same for all levels
- **Wavelength-dependent**: `α(λ) = α_0 × λ / λ_0` (attenuate higher frequencies to reduce noise)

### Attenuation

To prevent over-amplification (clipping, artifacts):

```
if |amplified[x,y,t]| > threshold:
    amplified[x,y,t] *= threshold / |amplified[x,y,t]|  # clamp
```

Or use soft attenuation:
```
amplified[x,y,t] *= (1 - exp(-|amplified[x,y,t]| / σ))  # soft clamp
```

## Breath Rate Estimation

Once we have the filtered signal, estimate breath rate via peak detection:

### Method 1: Time-Domain Peak Detection

1. Extract representative signal (e.g., average of torso region)
2. Find peaks in temporal signal using scipy.signal.find_peaks
3. Compute inter-peak intervals → breath period → BPM

```python
from scipy.signal import find_peaks

signal = filtered_pyramid[level][torso_mask].mean(axis=(0,1))  # (T,)
peaks, _ = find_peaks(signal, distance=fps/2)  # min 0.5s between peaks
intervals = np.diff(peaks) / fps  # seconds
breath_rate = 60 / intervals.mean()  # BPM
```

### Method 2: Frequency-Domain (FFT)

1. Compute FFT of temporal signal
2. Find dominant frequency in breathing band (0.1-0.7 Hz)
3. Convert to BPM

```python
from scipy.fft import fft, fftfreq

signal = filtered_pyramid[level][torso_mask].mean(axis=(0,1))
fft_vals = np.abs(fft(signal))
freqs = fftfreq(len(signal), 1/fps)

# Find peak in breathing band
mask = (freqs >= 0.1) & (freqs <= 0.7)
peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
breath_rate = peak_freq * 60  # BPM
```

## Parameters in Rickson UI

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| **Alpha (α)** | 0-50 | 10 | Amplification factor |
| **Low Freq** | 0.05-2 Hz | 0.1 Hz | Lower cutoff (6 BPM) |
| **High Freq** | 0.1-3 Hz | 0.7 Hz | Upper cutoff (42 BPM) |
| **Pyramid Levels** | 2-8 | 4 | Spatial decomposition depth |

**Scrubbable in UI:** Adjust in real-time and see immediate effect on:
- Amplified video output
- Breath rate estimate
- Breath cadence waveform

## Optimization Strategy

### CPU Reference (Current)

- Python + NumPy
- scipy.signal for IIR filters
- Straightforward to debug and validate

**Performance:** ~40-50 ms/frame @ 1920x1080 (too slow for real-time)

### GPU CUDA (Next)

Move heavy ops to GPU:

1. **Spatial pyramid**: CUDA kernel for Gaussian blur + downsample
2. **Temporal filter**: CUDA kernel for IIR (streaming, maintains state)
3. **Amplification**: Simple element-wise multiply on GPU
4. **Reconstruction**: CUDA kernel for pyramid collapse

**Expected performance:** ~2-5 ms/frame (10-20x speedup)

### RTX Compute Shader (Future)

Use RTX compute shaders for even tighter integration with rendering:

- Compute directly in Omniverse render graph
- Zero-copy between compute and render
- Full pipeline on GPU

**Expected performance:** ~1-2 ms/frame

## Challenges & Solutions

### Challenge 1: Temporal Delay

IIR filters need history → latency

**Solution:**
- Use shortest feasible filter order (N=2)
- Accept ~1-2 breath cycles for convergence
- Show "warming up" indicator in UI

### Challenge 2: Motion Artifacts

Camera motion or subject motion can dominate signal

**Solution:**
- Use pose tracking to segment athlete
- Apply EVM only to regions of interest (torso for breath)
- Stabilize camera or compute optical flow

### Challenge 3: Lighting Variations

Scene lighting changes can swamp breath signal

**Solution:**
- Normalize each pyramid level
- Use chrominance (color) rather than luminance for pulse
- High-pass filter to remove DC trends

### Challenge 4: Noise Amplification

α amplifies noise as well as signal

**Solution:**
- Lower α (trade off visibility)
- Denoise input (risk removing signal)
- Use wavelength-dependent attenuation
- Spatial averaging over torso region

## Validation

### Ground Truth

Compare against:
1. **Chest strap sensor** (respiratory rate monitor)
2. **Manual annotation** (count breaths in video)
3. **Synchronized CO₂ monitor**

### Metrics

- **Accuracy**: |estimated_BPM - ground_truth_BPM| < 2 BPM
- **Latency**: Time to first valid estimate < 3 breath cycles
- **Robustness**: Accuracy across lighting, motion, clothing

### Test Cases

1. **Static subject, controlled lighting**: Baseline
2. **Subject moving (BJJ drill)**: Motion robustness
3. **Varying lighting**: Lighting robustness
4. **Different clothing (gi vs. rash guard)**: Material robustness

## Example Output

**Input:** 30-second BJJ sparring clip @ 30fps, 1920x1080

**EVM Parameters:**
- α = 15
- Band: 0.15-0.6 Hz (9-36 BPM)
- Levels: 4

**Output:**
- Amplified video showing torso expansion/contraction
- Breath rate: 22.4 BPM (±0.8 BPM confidence interval)
- Breath cadence graph (time-domain signal)
- Spectrogram showing dominant breathing frequency

**Use in Training:**
- Athlete sees breath holding during submission attempts
- Coach sees breath rate spike during high-intensity drills
- Compare breath control between sparring rounds

## References

1. **Original EVM Paper:**
   Wu et al., "Eulerian Video Magnification for Revealing Subtle Changes in the World", ACM TOG (SIGGRAPH), 2012
   https://people.csail.mit.edu/mrub/papers/vidmag.pdf

2. **Follow-up (Phase-based):**
   Wadhwa et al., "Phase-based Video Motion Processing", ACM TOG (SIGGRAPH), 2013
   https://people.csail.mit.edu/nwadhwa/phase-video/

3. **Real-time Implementation:**
   Elgharib et al., "Video Magnification in Presence of Large Motions", CVPR 2015

4. **Breath Rate from Video:**
   Tarassenko et al., "Non-contact video-based vital sign monitoring", IEEE EMBC, 2014

5. **Code & Demos:**
   - MIT CSAIL EVM Matlab: http://people.csail.mit.edu/mrub/evm/
   - OpenCV Python examples: https://github.com/tbl3rd/Pyramids-with-OpenCV
