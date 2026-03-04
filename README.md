# RoX: An Experimental AR Voxel Engine

RoX is a custom-built **Augmented Reality voxel engine** that turns a standard webcam into a spatial computing sensor. Built entirely from scratch in Python, it bridges the gap between raw computer vision and hardware-accelerated 3D rendering.

![Python](https://img.shields.io/badge/Python-3.11.1-blue) ![OpenGL](https://img.shields.io/badge/OpenGL-ModernGL-green) ![Computer Vision](https://img.shields.io/badge/Computer_Vision-MediaPipe-orange)

### The Manifesto: Hardware Democracy
High-end spatial computing (VR/AR) is currently locked behind expensive, specialized hardware. RoX was born from a simple question: **Can we provide a high-fidelity AR experience to anyone with a basic laptop?**

By prioritizing sophisticated math and kinematic prediction over expensive sensors, RoX aims to democratize spatial interfaces. This isn't just a block-builder; it's a proof-of-concept for accessible, resourceful AR that can be used in schools or by hobbyists who don't have access to costly headsets.

---

### Demo: The Ghost Frame System in Action

To truly understand why RoX is different from standard AR filters, you have to see the **Kinematic Prediction Engine** at work. Raw computer vision often drops tracking during fast movements or occlusions.

RoX includes a built-in comparison tool that runs the raw MediaPipe feed side-by-side with the RoX Ghost Frame engine.

**Run the demo:**
```bash
python demo_rox_demo.py
```

*(This script processes a test video and generates a side-by-side comparison of tracking stability.)*

---

## Engineering Deep Dive

### Screen-to-World Raycasting

Mapping a 2D mouse or finger coordinate on a webcam feed to a 3D voxel requires reconstructing the entire graphics pipeline in reverse.

To find the world-space position:

$$
\mathbf{P}_{world,h} =
\left(\mathbf{M}_{proj} \cdot \mathbf{M}_{view}\right)^{-1}
\cdot
\mathbf{P}_{ndc}
$$

Since this result is in homogeneous coordinates, perform the perspective divide:

$$
\mathbf{P}_{world} =
\frac{\mathbf{P}_{world,h}.xyz}
{\mathbf{P}_{world,h}.w}
$$

---

### The Ghost Frame Hybrid Tracking

![AR Ghost Frame Generation](readme_assets/compare_ar_with_without_generation.gif)

The average velocity over the last $n$ frames:

$$
\vec{V}_{avg} =
\frac{1}{n}
\sum_{i=1}^{n}
\frac{\vec{P}_i - \vec{P}_{i-1}}
{f_i - f_{i-1}}
$$

Momentum decay with kinematic friction:

$$
\vec{V}_{t+1} =
\vec{V}_t \cdot K_{friction}
$$

---

## Features & Architecture

- **Dual-Engine Architecture:** Toggle between `3Drox.py` (perspective building) and `2Drox.py` (top-down design).  
- **State-Driven Interaction:** Uses a robust state machine (`HandActionState`) to handle gesture debouncing and pinch stability.  
- **ModernGL Pipeline:** Custom GLSL shaders handle real-time lighting and texture-array indexing for varied voxel types.

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aimisnotavailable/RoX.git
cd RoX
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the engine**

- **3D Mode**
```bash
python 3Drox.py
```

- **2D Mode**
```bash
python 2Drox.py
```

---