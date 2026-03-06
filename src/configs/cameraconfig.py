from scripts.arconfig import HEIGHT, WIDTH
from src.configs.importconfig import *
from src.configs.globalconfig import *

# camera
ASPECT_RATIO = WIDTH / HEIGHT 
FOV_DEG = 50
V_FOV = glm.radians(FOV_DEG)  # vertical FOV
H_FOV = 2 * math.atan(math.tan(V_FOV * 0.5) * ASPECT_RATIO)  # horizontal FOV
NEAR = 0.1
FAR = 2000.0
PITCH_MAX = glm.radians(89)

ZOOM_MIN = 0.25
ZOOM_MAX = 4.0
CHUNK_SPHERE_RADIUS = H_CHUNK_SIZE * math.sqrt(3)