import math
import cv2
import mediapipe as mp
import pygame
import sys
import random
import time
import glm

from collections import deque, namedtuple
from scripts.logger import get_logger_info
from screeninfo import get_monitors


# ------- WINDOW CONFIGURATION ------------------
HEIGHT = get_monitors()[0].height
WIDTH = get_monitors()[0].width
# -----------------------------------------------


# ---- AR CONFIG & UTILS --- #
# --- CONFIGURATION & INDICES ---
THUMB_TIP_IDX   = 4
INDEX_TIP_IDX   = 8
MIDDLE_MCP_IDX  = 9
WRIST_IDX       = 0

HISTOGRAM_SIZE     = 5
PINCH_ON_THRESH    = 0.15    # normalized units 
PINCH_OFF_THRESH   = 0.20
PINCH_FRAMES_REQ   = 3       # debounce frames

# Keep these if you still need them
SCALE_SIZE    = 8
ATTACK_COOLDOWN = 20

# ------------------

# 2D CONFIGS -----------------------
BLOCK_SIZE = 16

# TUNABLES
PLACEMENT_COOLDOWN = 0.1
PLACEMENT_MIN_MOVE = 4.0
PINCH_STABLE_TIME = 0.06
MIN_SCREEN_BLOCK_SIZE = 2
ZOOM_MIN = 0.25
ZOOM_MAX = 4.0

# Block visuals
BLOCK_VISUAL_SCALE = 1.0
BLOCK_VISUAL_SCALE_ALLOW_OVERLAP = False

# Finger smoothing
FINGER_EMA_ALPHA = 0.45

# Camera pan interpolation
DEADZONE_PX = 6
SMOOTH_TAU = 0.06
INSTANT_STOP = False
MAX_STEP_WORLD = 10000.0

# Brightside lighting defaults
BRIGHTSIDE_DEFAULT = True
LIGHT_DIR = (-0.6, -0.4)        # screen-space light direction (x,y)
HIGHLIGHT_INTENSITY = 0.28
SHADOW_INTENSITY = 0.12
