```markdown
# RoX: An Experimental AR Voxel Engine

RoX is a custom-built **Augmented Reality voxel engine** that turns a standard webcam into a spatial computing sensor. Built entirely from scratch in Python, it bridges the gap between raw computer vision and hardware-accelerated 3D rendering.

![Python](https://img.shields.io/badge/Python-3.11.1-blue) ![OpenGL](https://img.shields.io/badge/OpenGL-ModernGL-green) ![Computer Vision](https://img.shields.io/badge/Computer_Vision-MediaPipe-orange)

### The Manifesto: Hardware Democracy
High-end spatial computing (VR/AR) is currently locked behind expensive, specialized hardware. RoX was born from a simple question: **Can we provide a high-fidelity AR experience to anyone with a basic laptop?**

By prioritizing sophisticated math and kinematic prediction over expensive sensors, RoX aims to democratize spatial interfaces. This isn't just a block-builder; it's a proof-of-concept for accessible, resourceful AR that can be used in schools or by hobbyists who don't have access to costly headsets.

**Zero-Cost Architecture:** RoX intentionally avoids expensive algorithmic band-aids (like CLAHE for contrast fixing) that would overheat lower-end CPUs. Instead, it relies on zero-cost logic gates—like the "Mercy Kill" static friction cutoff—to correct tracking errors, ensuring the computer vision thread stays well under its 33ms budget.

---

### Demo: The Ghost Frame System in Action

![AR Ghost Frame Generation](readme_assets/compare_ar.gif)

To truly understand why RoX is different from standard AR filters, you have to see the **Kinematic Prediction Engine** at work. Raw computer vision often drops tracking during fast movements or occlusions.

RoX includes a built-in comparison tool that runs the raw MediaPipe feed side-by-side with the RoX Ghost Frame engine.

**Run the demo:**
```bash
python demo_rox.py --input path/to/video.mp4 --output results/comparison.gif --fps 20

```

*(This script processes a test video and generates a side-by-side comparison of tracking stability.)*

---

## Engineering Deep Dive

### Screen-to-World Raycasting

Mapping a 2D mouse or finger coordinate on a webcam feed to a 3D voxel requires reconstructing the entire graphics pipeline in reverse.

To find the world-space position:

$$\mathbf{P}_{world,h} =
\left(\mathbf{M}_{proj} \cdot \mathbf{M}_{view}\right)^{-1}
\cdot
\mathbf{P}_{ndc}$$

Since this result is in homogeneous coordinates, perform the perspective divide:

$$\mathbf{P}_{world} =
\frac{\mathbf{P}_{world,h}.xyz}
{\mathbf{P}_{world,h}.w}$$

---

### The Ghost Frame Hybrid Tracking System

Raw computer vision often drops tracking during fast movements, motion blur, or severe backlight. RoX solves this using a multi-tiered predictive physics engine, generating "Ghost Frames" to seamlessly bridge the gap between actual hardware sensor updates.

**Layer 1: Optical Flow (Lucas-Kanade)**
When MediaPipe drops a frame, RoX does not immediately guess the hand's position. Instead, it relies on OpenCV's Lucas-Kanade optical flow algorithm to physically track the raw pixel shifts of the hand's silhouette.

This grounds the prediction in relative physical reality rather than absolute coordinate guessing, bypassing issues caused by blown-out contrast or harsh lighting.

**Layer 2: Kinematic Fallback & Friction**
If optical flow completely fails (e.g., a pure white washed-out frame), the engine falls back to pure kinematics. It calculates the initial velocity before the drop and applies continuous kinematic friction until the physical hand is found or static friction halts the movement.

The average velocity over the last $n$ frames:

$$\vec{V}_{avg} =
\frac{1}{n}
\sum_{i=1}^{n}
\frac{\vec{P}_i - \vec{P}_{i-1}}
{f_i - f_{i-1}}$$

Momentum decay with kinematic friction:

$$\vec{V}_{t+1} =
\vec{V}_t \cdot K_{friction}$$

---

### State Governance & Hysteresis

Tracking the position of a lost hand is only half the battle. If an engine drops the user's input state (like a "pinch" or "click") during a dropped frame, continuous actions like dragging and extruding voxels become impossible.

RoX implements **State Governance** for its Ghost Frames. When the physical camera drops a frame, the optical flow system inherits the last known physical state of the hand.

If the user was pinching, the Ghost Frame holds that pinch through the camera stutter. This state hysteresis is tightly governed by a Time-To-Live (TTL) kill-switch, preventing permanent "zombie" inputs while smoothing over micro-stutters, making a standard 30fps webcam feel like a continuous high-end sensor.

---

## Features & Architecture

* **Dual-Engine Architecture:** Toggle between `3Drox.py` (perspective building) and `2Drox.py` (top-down design).
* **State-Driven Interaction:** Uses a robust state machine (`HandActionState`) to handle gesture debouncing and pinch stability.
* **ModernGL Pipeline:** Custom GLSL shaders handle real-time lighting and texture-array indexing for varied voxel types.

---

## Installation

1. **Clone the repository**

```bash
git clone [https://github.com/Aimisnotavailable/RoX.git](https://github.com/Aimisnotavailable/RoX.git)
cd RoX

```

2. **Install dependencies**

```bash
pip install -r requirements.txt

```

3. **Run the engine**

* **3D Mode**

```bash
python 3Drox.py

```

* **2D Mode**

```bash
python 2Drox.py

```

```

```