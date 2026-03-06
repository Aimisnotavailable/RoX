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