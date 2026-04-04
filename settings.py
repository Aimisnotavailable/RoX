from numba import njit
import numpy as np
import glm
import math
import json
import pygame
from pathlib import Path
from scripts.logger import get_logger_info
import cv2

RENDER_DISTANCE = 8

# File format
CHUNK_FILE_FORMAT = ".json"
CHUNK_FILE_BASE_DIR =  Path(f'world_data/chunks')

# OpenGL settings
MAJOR_VER, MINOR_VER = 3, 3
DEPTH_SIZE = 24
NUM_SAMPLES = 1  # antialiasing

# resolution
WIN_RES = glm.vec2(1600, 900)

# world generation
SEED = 11405

# ray casting
MAX_RAY_DIST = 6

# chunk
CHUNK_SIZE = 48
H_CHUNK_SIZE = CHUNK_SIZE // 2
CHUNK_AREA = CHUNK_SIZE * CHUNK_SIZE
CHUNK_VOL = CHUNK_AREA * CHUNK_SIZE
CHUNK_SPHERE_RADIUS = H_CHUNK_SIZE * math.sqrt(3)

# world
WORLD_W, WORLD_H = 15, 15
WORLD_D = WORLD_W
WORLD_AREA = WORLD_W * WORLD_D
WORLD_VOL = WORLD_AREA * WORLD_H

# world center
CENTER_XZ = WORLD_W * H_CHUNK_SIZE
CENTER_Y = WORLD_H * H_CHUNK_SIZE

# camera
ASPECT_RATIO = WIN_RES.x / WIN_RES.y
FOV_DEG = 50
V_FOV = glm.radians(FOV_DEG)  # vertical FOV
H_FOV = 2 * math.atan(math.tan(V_FOV * 0.5) * ASPECT_RATIO)  # horizontal FOV
NEAR = 0.1
FAR = 2000.0
PITCH_MAX = glm.radians(89)
ZOOM_MIN = 0.2
ZOOM_MAX = 5

# player
MAX_PLAYER_SPEED = 0.3
PLAYER_SPEED = 0.05
PLAYER_ROT_SPEED = 0.03
# PLAYER_POS = glm.vec3(CENTER_XZ, WORLD_H * CHUNK_SIZE, CENTER_XZ)
PLAYER_POS = glm.vec3(CENTER_XZ, CHUNK_SIZE, CENTER_XZ)
MOUSE_SENSITIVITY = 0.002


# colors
BG_COLOR = glm.vec3(0.58, 0.83, 0.99)

# textures
SAND = 1
GRASS = 2
DIRT = 3
STONE = 4
SNOW = 5
LEAVES = 6
WOOD = 7

# terrain levels
SNOW_LVL = 54
STONE_LVL = 49
DIRT_LVL = 40
GRASS_LVL = 8
SAND_LVL = 7

# tree settings
TREE_PROBABILITY = 0.02
TREE_WIDTH, TREE_HEIGHT = 4, 8
TREE_H_WIDTH, TREE_H_HEIGHT = TREE_WIDTH // 2, TREE_HEIGHT // 2

# water
WATER_LINE = 5.6
WATER_AREA = 5 * CHUNK_SIZE * WORLD_W

# cloud
CLOUD_SCALE = 25
CLOUD_HEIGHT = WORLD_H * CHUNK_SIZE * 2

# WORLD
INTERACTION_MODE = ['REMOVE', 'ADD', 'GRAB', 'VIEWING']
INTERACTION_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# -------------------- DEPTH INTEGRATION --------------------
# AR depth gesture settings

Z_NEAR = 0.3          # closer than this -> remove mode
Z_FAR = 0.7           # farther than this -> add mode
Z_HYSTERESIS = 0.05   # dead zone to avoid flickering
Z_HOLD_FRAMES = 5     # frames to hold depth before switching mode
Z_DRAG_SPEED_MIN = 0.3
Z_DRAG_SPEED_MAX = 2.0
Z_ROTATION_MIN = 0.5
Z_ROTATION_MAX = 2.0

LEFT_ZOOM_SENSITIVITY = 2.0      # how much depth delta affects world scale
LEFT_ZOOM_MIN = 0.1
LEFT_ZOOM_MAX = 10.0

RIGHT_BRUSH_SENSITIVITY = 1.5    # how much depth delta affects brush multiplier
BRUSH_MULT_MIN = 2
BRUSH_MULT_MAX = 3.0

BOTH_HANDS_HOLD_TIME = 1.0       # seconds to hold both pinched to trigger action


# RADIAL MENUS

# Block selection menu (without grab size)
BLOCKS_MENU = [
    {"name": "SAND",   "voxel_id": 1, "color": (230,210,180)},
    {"name": "GRASS",  "voxel_id": 2, "color": (100,200,100)},
    {"name": "DIRT",   "voxel_id": 3, "color": (140,100,70)},
    {"name": "STONE",  "voxel_id": 4, "color": (160,160,170)},
    {"name": "SNOW",   "voxel_id": 5, "color": (240,240,255)},
    {"name": "LEAVES", "voxel_id": 6, "color": (80,160,80)},
    {"name": "WOOD",   "voxel_id": 7, "color": (180,140,100)},
    {"name": "BACK",   "action": "back", "color": (100,100,100)}
]

# Grab size submenu
GRAB_SIZE_MENU = [
    {"name": "SIZE 1", "size": 1, "color": (200,200,200)},
    {"name": "SIZE 3", "size": 3, "color": (200,200,200)},
    {"name": "SIZE 5", "size": 5, "color": (200,200,200)},
    {"name": "BACK",   "action": "back", "color": (100,100,100)}
]


WORLD_GEN_PARAMS = {
    "sphere":   {"radius": 80},
    "torus":    {"R": 80, "r": 20},
    "cube":     {"half_size": 80},
    # "cylinder": {"radius": 40, "height": 80, "axis": "y"},
    # "sinewave": {"amplitude": 15, "wavelength": 30},
    # "wave":     {"amplitude": 12, "wavelength_x": 30, "wavelength_z": 30},
    # "hill":     {"radius": 80, "height": 50},
    # "pyramid":  {"half_base": 60, "height": 80},
    "goursat":  {"scale": 30},
    "steinmetz":{"radius": 80},
    "heart":    {"scale": 40},
    "spiked_sphere": {"radius": 80, "spikes": 8, "amplitude": 0.3},
    "rounded_octahedron": {"size": 80, "exponent": 4},
    "mobius":   {"R": 40, "width": 20, "thickness": 3}
}

WORLD_GEN_MENU = [
    {"name": "SPHERE",   "color": (100,200,200), "type": "sphere",   "params": WORLD_GEN_PARAMS["sphere"]},
    {"name": "TORUS",    "color": (200,100,200), "type": "torus",    "params": WORLD_GEN_PARAMS["torus"]},
    {"name": "CUBE",     "color": (200,200,100), "type": "cube",     "params": WORLD_GEN_PARAMS["cube"]},
    # {"name": "CYLINDER", "color": (100,200,100), "type": "cylinder", "params": WORLD_GEN_PARAMS["cylinder"]},
    # {"name": "SINEWAVE", "color": (200,150,100), "type": "sinewave", "params": WORLD_GEN_PARAMS["sinewave"]},
    # {"name": "WAVE",     "color": (150,200,150), "type": "wave",     "params": WORLD_GEN_PARAMS["wave"]},
    # {"name": "HILL",     "color": (180,140,100), "type": "hill",     "params": WORLD_GEN_PARAMS["hill"]},
    # {"name": "PYRAMID",  "color": (200,180,120), "type": "pyramid",  "params": WORLD_GEN_PARAMS["pyramid"]},
    {"name": "BACK",     "color": (100,100,100), "action": "back"},
]

# Top‑level main menu
MAIN_MENU = [
    {"name": "BLOCKS",   "submenu": BLOCKS_MENU,    "color": (200,200,200)},
    {"name": "GRAB SIZE","submenu": GRAB_SIZE_MENU, "color": (200,200,200)},
    {"name": "WORLD",    "submenu": WORLD_GEN_MENU, "color": (200,200,200)},
    {"name": "EXIT",     "action": "exit",          "color": (150,150,150)}
]