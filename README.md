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

Mapping a 2D mouse or finger coordinate on a webcam feed to a 3D voxel requires reconstructing the entire graphics pipeline in reverse. To find the world-space position \( \mathbf{P}_{\text{world}} \), we transform Normalized Device Coordinates \( \mathbf{P}_{\text{ndc}} \) through the inverse of the View–Projection matrix:

\[
\mathbf{P}_{\text{world},h} = \bigl(\mathbf{M}_{\text{proj}} \cdot \mathbf{M}_{\text{view}}\bigr)^{-1} \cdot \mathbf{P}_{\text{ndc}}
\]

Since this result is in homogeneous coordinates, perform the perspective divide to reach the final 3D world coordinate:

\[
\mathbf{P}_{\text{world}} = \frac{\mathbf{P}_{\text{world},h}.xyz}{\mathbf{P}_{\text{world},h}.w}
\]

### The Ghost Frame Hybrid Tracking

Raw computer vision is fragile. Motion blur and occlusion often cause tracking to drop. RoX treats the hand as a physical object with momentum. When tracking is lost, the engine calculates the average velocity over the last \(n\) frames:

\[
\vec{V}_{\text{avg}} = \frac{1}{n} \sum_{i=1}^{n} \frac{\vec{P}_i - \vec{P}_{i-1}}{f_i - f_{i-1}}
\]

The engine then injects *Ghost Frames* that advance the hand's skeletal position along this trajectory. To ensure a natural feel, a kinematic friction coefficient \(K_{\text{friction}}\) is applied to decay the momentum over time:

\[
\vec{V}_{t+1} = \vec{V}_t \cdot K_{\text{friction}}
\]

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

## Troubleshooting: Images and Math Rendering

- **Badges / images not rendering:** Ensure your Markdown renderer allows external images. If you are viewing the file locally, some viewers block remote images; try opening the file in GitHub or enable remote images in your viewer.
- **Math not rendering:** Math blocks use LaTeX. If your renderer does not support MathJax or KaTeX, formulas will appear as plain text. On GitHub, use fenced math with `$$ ... $$` for display math; many static viewers require enabling math rendering or using a plugin.
- **Local preview tips:** Use a Markdown viewer that supports MathJax/KaTeX (e.g., VS Code with a math extension) or view the file on GitHub to see badges and LaTeX rendered correctly.

---

If you still see visual artifacts or rendering issues after these fixes, check your Markdown viewer settings (remote images enabled, math rendering enabled) and confirm your GPU drivers and ModernGL installation are up to date.