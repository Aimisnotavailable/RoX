# RoX

RoX is a custom-built Augmented Reality voxel engine that enables users to build 2D and 3D structures using hand gestures.

Unlike standard AR projects that rely on game engines like Unity or frameworks like ARCore, RoX features a **custom rendering engine written from scratch** using Python, ModernGL (OpenGL), and OpenCV. It bridges raw computer vision data with a custom graphics pipeline to solve complex "Screen-to-World" coordinate mapping problems.

![Python](https://img.shields.io/badge/Python-3.11.1-blue) ![OpenGL](https://img.shields.io/badge/OpenGL-ModernGL-green) ![Computer Vision](https://img.shields.io/badge/Computer_Vision-MediaPipe-orange)

## Features

* **Hand Tracking:** Real-time skeletal tracking using MediaPipe with custom smoothing.
* **Gesture Recognition:** Custom logic for pinch detection, open-palm detection, and "Ghost Frames" to handle tracking loss.
* **Custom 3D Engine:** A lightweight OpenGL renderer built on ModernGL (VBOs, VAOs, Shaders).
* **Dual Modes:** Includes both a 2D canvas (`2Drox.py`) and a fully 3D voxel world (`3Drox.py`).
* **Voxel System:** Minecraft-style block placement with texture arrays and UV mapping.
* **Occlusion Culling:** Optimized rendering that only draws visible block faces to maintain high FPS.
* **Hybrid Pipeline:** Integration of OpenCV (input processing) and Pygame/OpenGL (rendering) within a single loop.

## Technical Stack

This project integrates three distinct systems into a unified realtime application:

1.  **Input Layer (OpenCV + MediaPipe):** Captures webcam feed and extracts 21-point hand landmarks in 2D screen space.
2.  **Math Layer (NumPy + GLM):** Converts 2D hand coordinates into 3D raycasts (Screen-to-World mapping) to determine interaction points in 3D space.
3.  **Render Layer (ModernGL + GLSL):** A custom forward renderer using GLSL shaders for lighting, texture mapping, and wireframe highlights.

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

    **requirements.txt:**
    ```text
    pygame-ce
    mediapipe==0.10.14
    opencv-python
    numpy
    pandas
    screeninfo
    colorama
    moderngl
    pyglm
    Pillow
    ```

3.  **Run the Engine:**
    * **For 3D Voxel Mode:**
        ```bash
        python 3Drox.py
        ```
    * **For 2D Canvas Mode:**
        ```bash
        python 2Drox.py
        ```

## Controls & Gestures

| Hand Action | Result |
| :--- | :--- |
| **Right Hand Point/Pinch** | **Build Mode** (Place Block) |
| **Right Hand Open Palm** | **Delete Mode** (Remove Block) |
| **Left Hand Pinch + Drag** | **Rotate Camera** (Orbit View) |
| **Two Hands Pinch + Pull** | **Zoom In/Out** |
| **Keyboard 'M'** | Switch Camera Mode (FPS / RTS) |
| **Keyboard 'Esc'** | Quit |

## Engineering Highlights

### The "Ghost Frame" System
To combat the high latency and occasional dropout of webcam hand tracking, RoX implements a **Ghost Frame** prediction algorithm. If the camera loses sight of a hand (due to motion blur or occlusion), the engine uses the last known velocity vectors to simulate the hand's position for several frames, ensuring smooth input continuity without jitter.

### Custom Raycasting
Since this engine does not use Unity, there is no built-in `ScreenPointToRay`. I implemented a custom raycaster using the Inverse View-Projection Matrix:
$$ P_{world} = M^{-1}_{view} \cdot M^{-1}_{proj} \cdot P_{ndc} $$
This allows 2D screen coordinates from the webcam to interact accurately with 3D voxel data.

### Texture Arrays
To render thousands of blocks efficiently in Python, the engine uses **OpenGL Texture Arrays**. Instead of binding a different texture for every block type (which causes significant CPU overhead), all textures are loaded into a single 3D texture array, and the shader selects the correct layer based on the block ID.

## License
MIT License