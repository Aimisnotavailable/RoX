# RoX: An Experimental AR Voxel Engine

RoX is a custom-built, experimental Augmented Reality voxel engine. It allows users to build 2D and 3D structures in real-time using nothing but their webcam and hand gestures.

Rather than relying on heavy game engines like Unity or pre-packaged AR frameworks, RoX was built entirely from scratch in Python. It serves as a personal exploration into bridging raw computer vision data with a custom graphics pipeline to solve spatial mapping and real-time interaction challenges.

![Python](https://img.shields.io/badge/Python-3.11.1-blue) ![OpenGL](https://img.shields.io/badge/OpenGL-ModernGL-green) ![Computer Vision](https://img.shields.io/badge/Computer_Vision-MediaPipe-orange)

## About The Project

This project started as a "tiny experiment" to see if a playable AR Minecraft-style interface could be built natively in Python. It evolved into a dual-engine architecture featuring:
* **3Drox.py**: A 3D voxel world with perspective projection, custom shaders, and depth testing.
* **2Drox.py**: A specialized 2D engine for top-down grid manipulation and canvas building.

It utilizes **MediaPipe** for skeletal landmark extraction, **OpenCV** for optical flow tracking, and **ModernGL** for hardware-accelerated rendering.

---

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Aimisnotavailable/RoX.git](https://github.com/Aimisnotavailable/RoX.git)
    cd RoX
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Engine:**
    * **3D Mode:** `python 3Drox.py`
    * **2D Mode:** `python 2Drox.py`

---

## Engineering Highlights

### State-Driven Interaction
Raw webcam data is incredibly jittery. To make placing blocks actually feel good, RoX uses an Exponential Moving Average (EMA) smoothing layer combined with a custom state machine (`HandActionState`). This handles gesture debouncing, pinch stability, and movement deltas, requiring deliberate human intent to trigger actions rather than accidental twitches.

### Screen-to-World Raycasting
To figure out where a user is "pinching" in the 3D world, RoX maps 2D webcam coordinates to 3D space by manually reconstructing the view-projection pipeline. The transformation from Normalized Device Coordinates (NDC) back to World Space is calculated as:

$$ P_{world} = (M_{proj} \cdot M_{view})^{-1} \cdot P_{ndc} $$

### The "Ghost Frame" Hybrid Tracking System

![AR Ghost Frame Generation](readme_assets/compare_ar_with_without_generation.gif)

A major challenge in building AR using raw computer vision is frame dropping. Fast hand movements cause motion blur, resulting in the tracking instantly failing. 

To prevent the engine from stuttering, RoX implements a custom **Sensor Fusion / Hybrid Tracker**:
1. **The Anchor (MediaPipe):** Acts as the ground-truth anatomical position.
2. **The Micro-Tracker (Optical Flow):** When MediaPipe drops out due to blur, OpenCV's Lucas-Kanade optical flow takes over, tracking the exact skin pixels of the joints frame-by-frame.
3. **Kinematic Prediction Engine:** If the hand leaves the frame entirely, the engine calculates the average linear and angular velocities from the last known good frames:

$$\vec{V}_{linear} = \frac{\sum (P_{new} - P_{old})}{\Delta f}$$
$$V_{angular} = \frac{\sum (\theta_{new} - \theta_{old})}{\Delta f}$$

The engine then seamlessly injects "Ghost Frames" into the pipeline, advancing the skeletal wireframe along this trajectory. A kinematic friction multiplier decays the momentum gracefully ($$\vec{V}_{t+1} = \vec{V}_t \cdot K_{friction}$$), making the hand glide to a stop instead of flying off into infinity.