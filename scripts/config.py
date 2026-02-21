import math
import cv2
import mediapipe as mp
import pygame
import sys

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
