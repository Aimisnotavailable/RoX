# RoX

RoX is a custom-built Augmented Reality voxel engine that enables users to build 2D and 3D structures using hand gestures. 

Unlike standard AR projects that rely on high-level engines like Unity or frameworks like ARCore, RoX features a **custom rendering engine built from the ground up** using Python 3.11, ModernGL (OpenGL 3.3 Core), and OpenCV. It bridges raw computer vision data with a custom graphics pipeline to solve complex spatial mapping and real-time interaction challenges.

![Python](https://img.shields.io/badge/Python-3.11.1-blue) ![OpenGL](https://img.shields.io/badge/OpenGL-ModernGL-green) ![Computer Vision](https://img.shields.io/badge/Computer_Vision-MediaPipe-orange)

## Features

* **Real-time Hand Tracking:** Uses MediaPipe for skeletal landmark extraction with a custom Exponential Moving Average (EMA) smoothing layer.
* **State-Driven Interaction:** A robust state machine (`HandActionState`) handles gesture debouncing, pinch stability, and movement deltas.
* **Dual-Engine Architecture:** * `3Drox.py`: A high-performance 3D voxel world with perspective projection and depth testing.
    * `2Drox.py`: A specialized 2D engine for top-down grid manipulation and canvas building.
* **Optimized Rendering:** Implements VBO/IBO batching, GLSL shaders, and texture arrays to minimize draw calls.
* **Dynamic Viewports:** Supports both FPS (First-Person) and RTS (Top-Down) camera controllers.

## Technical Stack

* **Language:** Python 3.11.1
* **Graphics:** ModernGL (OpenGL 3.3 Core Profile)
* **Input/CV:** OpenCV, MediaPipe
* **Math:** PyGLM (OpenGL Mathematics for Python), NumPy
* **Windowing:** Pygame-ce (Community Edition)

## Installation

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

## Engineering Highlights

### Screen-to-World Raycasting
RoX maps 2D webcam coordinates to 3D space by manually reconstructing the view-projection pipeline. The transformation from Normalized Device Coordinates (NDC) back to World Space is calculated as:

$$ P_{world} = (M_{proj} \cdot M_{view})^{-1} \cdot P_{ndc} $$

We then perform a perspective divide by $w$ to find the precise intersection ray for voxel placement.

### The "Ghost Frame" System
To mitigate tracking drops common in computer vision, RoX uses a velocity-based prediction system. If a hand is occluded, the engine projects its trajectory based on previous frame momentum, preventing "jitter" and ensuring the UI remains responsive during fast movements.

### Texture Array Batching
To render varied block types (Sand, Grass, Stone, etc.) without the overhead of switching textures, the engine utilizes a **Sampler2DArray**. This allows the Fragment Shader to index specific textures in a single draw call based on an integer ID passed through the vertex data.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.