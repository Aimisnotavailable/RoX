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
THUMB_TIP_IDX   = 4
INDEX_TIP_IDX   = 8
INDEX_MCP_IDX   = 5      # Index finger MCP (base)
MIDDLE_MCP_IDX  = 9      # Middle finger MCP
PINKY_MCP_IDX   = 17     # Pinky finger MCP
WRIST_IDX       = 0

# --- History and pinch detection ---
HISTOGRAM_SIZE     = 10
PINCH_ON_THRESH    = 0.30    # Triggers a click earlier
PINCH_OFF_THRESH   = 0.45    # Releases the click smoothly
PINCH_FRAMES_REQ   = 3       # Debounce frames