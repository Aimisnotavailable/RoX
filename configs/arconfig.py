# arconfig.py
import math
import cv2
import mediapipe as mp
import pygame
import sys
import random
import time
import glm
import moderngl
from PIL import Image

from collections import deque, namedtuple
from scripts.logger import get_logger_info
from screeninfo import get_monitors


# ------- WINDOW CONFIGURATION ------------------
HEIGHT = get_monitors()[0].height
WIDTH = get_monitors()[0].width
# -----------------------------------------------


# ---- AR CONFIG & UTILS --- #
# --- Hand landmark indices (MediaPipe) ---
WRIST_IDX       = 0
THUMB_CMC_IDX   = 1
THUMB_MCP_IDX   = 2
THUMB_IP_IDX    = 3
THUMB_TIP_IDX   = 4
INDEX_MCP_IDX   = 5
INDEX_PIP_IDX   = 6
INDEX_DIP_IDX   = 7
INDEX_TIP_IDX   = 8
MIDDLE_MCP_IDX  = 9
MIDDLE_PIP_IDX  = 10
MIDDLE_DIP_IDX  = 11
MIDDLE_TIP_IDX  = 12
RING_MCP_IDX    = 13
RING_PIP_IDX    = 14
RING_DIP_IDX    = 15
RING_TIP_IDX    = 16
PINKY_MCP_IDX   = 17
PINKY_PIP_IDX   = 18
PINKY_DIP_IDX   = 19
PINKY_TIP_IDX   = 20

# --- History and pinch detection ---
HISTOGRAM_SIZE     = 10
PINCH_ON_THRESH    = 0.25   # was 0.30 – require closer fingers to trigger
PINCH_OFF_THRESH   = 0.40   # was 0.45 – release earlier
PINCH_FRAMES_REQ   = 3       # Debounce frames
PINCH_HOLD_FRAMES  = 15      # Frames to trigger hold action