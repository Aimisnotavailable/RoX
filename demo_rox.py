# render_compare_from_video.py
"""
Compare AR overlays side-by-side:
Left  = AR output (as produced in ar.ar_data, includes generated ghost frames)
Right = AR output but showing only REAL frames (suppress generated ghosts)
Saves a looping GIF, drops the final captured frame, and includes minimal indicators and logs.
"""

import cv2
import imageio
import numpy as np
import pygame
import sys
import time
from scripts.ar import AR
from datetime import datetime
import math

# Config
VIDEO_FILE = "demo/demo.mp4"
OUT_GIF = "demo/compare_ar_with_without_generation.gif"
WIDTH, HEIGHT = 640, 480
SCALE = 0.6
FPS = 15
BORDER = 4

# Colors
COLOR_LEFT = (255, 255, 255)    # left panel (AR with generation)
COLOR_RIGHT = (200, 200, 200)   # right panel (real-only)
COLOR_BORDER = (30, 30, 30)
COLOR_TITLE = (240, 240, 240)
COLOR_BG = (10, 10, 10)
COLOR_GHOST_TINT = (120, 120, 255, 40)
COLOR_TEXT = (220, 220, 220)
COLOR_DIAG = (255, 200, 0)
COLOR_WRIST_MARK = (255, 0, 0)
CONN_COLOR = (0, 200, 255)
INDICATOR_REAL = (0, 200, 0)
INDICATOR_GHOST = (160, 80, 200)

# Helpers --------------------------------------------------------------------

def safe_get_position_pts_from_ar_data(ar_data, label):
    """Return a list of (x,y) or None from ar_data POSITION_DATA for the given label."""
    pts = ar_data.get('POSITION_DATA', {}).get(label)
    if not pts:
        return []
    safe = []
    for p in pts:
        if not p:
            safe.append(None)
            continue
        try:
            x = int(round(p[0])); y = int(round(p[1]))
            safe.append((x, y))
        except Exception:
            safe.append(None)
    return safe

def draw_connections(surf, pts, color, width=2):
    """Draw hand connections best-effort using MediaPipe HAND_CONNECTIONS."""
    if not pts:
        return
    try:
        from mediapipe.python.solutions.hands import HAND_CONNECTIONS
        conns = HAND_CONNECTIONS
    except Exception:
        conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,9),(9,13),(13,17),(17,0)]
    for a, b in conns:
        if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
            pygame.draw.line(surf, color, pts[a], pts[b], width)

def draw_points(surf, pts, color, radius=3):
    if not pts:
        return
    for p in pts:
        if p:
            pygame.draw.circle(surf, color, p, radius)

def frame_rgb_to_surface(frame_rgb):
    """Convert HxWx3 RGB numpy array to a pygame Surface."""
    try:
        surf = pygame.image.frombuffer(frame_rgb.tobytes(), (frame_rgb.shape[1], frame_rgb.shape[0]), "RGB").convert()
    except Exception:
        surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)).convert()
    return surf

def surf_to_frame(surf, scale=1.0):
    """Convert a pygame Surface to an HxWx3 RGB numpy array suitable for imageio."""
    arr = pygame.surfarray.array3d(surf).swapaxes(0, 1)
    if scale != 1.0:
        new_w = int(arr.shape[1] * scale); new_h = int(arr.shape[0] * scale)
        tmp = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        tmp = pygame.transform.smoothscale(tmp, (new_w, new_h))
        arr = pygame.surfarray.array3d(tmp).swapaxes(0, 1)
    return arr.astype(np.uint8).copy()

# Main -----------------------------------------------------------------------

def main():
    pygame.init(); pygame.font.init()
    screen = pygame.display.set_mode((WIDTH * 2 + BORDER * 3, HEIGHT + BORDER * 3))
    clock = pygame.time.Clock()

    # AR pipeline (authoritative + ghost pruning)
    ar = AR((WIDTH, HEIGHT))

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print("Failed to open", VIDEO_FILE)
        return

    frames = []
    frame_idx = 0
    running = True

    # fonts
    try:
        title_font = pygame.font.SysFont("Arial", 18)
        small_font = pygame.font.SysFont("Arial", 14)
        diag_font = pygame.font.SysFont("Consolas", 12)
    except Exception:
        title_font = small_font = diag_font = None

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Update AR pipeline (it runs its own MediaPipe internally)
        ar_data = ar.update(frame)

        # Prepare a resized RGB frame for backgrounds
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(rgb, (WIDTH, HEIGHT))

        # Background surfaces (fresh each frame)
        bg_left_surf = frame_rgb_to_surface(frame_rgb)
        bg_right_surf = frame_rgb_to_surface(frame_rgb)

        # Compose left (with generation) and right (real-only) surfaces
        left_surf = pygame.Surface((WIDTH, HEIGHT))
        left_surf.blit(bg_left_surf, (0, 0))
        right_surf = pygame.Surface((WIDTH, HEIGHT))
        right_surf.blit(bg_right_surf, (0, 0))

        # ---- Left: Render directly from ar.ar_data (with generated frames) ----
        for label in ("LEFT", "RIGHT"):
            pts_left = safe_get_position_pts_from_ar_data(ar_data, label)
            frame_type = ar_data.get('FRAME_TYPE', {}).get(label, "REAL")
            # ghost tint if ghost
            draw_points(left_surf, pts_left, COLOR_LEFT, radius=3)
            draw_connections(left_surf, pts_left, CONN_COLOR, width=2)
            # wrist marker (index 0)
            try:
                if len(pts_left) > 0 and pts_left[0]:
                    pygame.draw.circle(left_surf, COLOR_WRIST_MARK, pts_left[0], 6)
            except Exception:
                pass
            # small per-hand label and indicator
            try:
                y_off = 18 if label == "LEFT" else 36
                left_surf.blit(small_font.render(f"{label}: {frame_type}", True, COLOR_TEXT), (6, y_off))
                ind_color = INDICATOR_REAL if frame_type == "REAL" else INDICATOR_GHOST
                pygame.draw.circle(left_surf, ind_color, (WIDTH - 18, y_off + 6), 6)
            except Exception:
                pass

        # ---- Right: Render only REAL frames from ar.ar_data (suppress ghosts) ----
        for label in ("LEFT", "RIGHT"):
            frame_type = ar_data.get('FRAME_TYPE', {}).get(label, "REAL")
            if frame_type == "REAL":
                pts_right = safe_get_position_pts_from_ar_data(ar_data, label)
                draw_points(right_surf, pts_right, COLOR_RIGHT, radius=3)
                draw_connections(right_surf, pts_right, CONN_COLOR, width=2)
                # wrist marker
                try:
                    if len(pts_right) > 0 and pts_right[0]:
                        pygame.draw.circle(right_surf, COLOR_WRIST_MARK, pts_right[0], 6)
                except Exception:
                    pass
                shown_type = "REAL"
            else:
                # suppress generated ghost: show nothing for this hand
                pts_right = []
                shown_type = "SUPPRESSED_GHOST"
                # debug log: if POSITION_DATA still contains points while FRAME_TYPE is GHOST, log it
                pos = ar_data.get('POSITION_DATA', {}).get(label)
                if pos and any(p for p in pos if p):
                    print(f"[DEBUG] frame {frame_idx} {label}: FRAME_TYPE=GHOST but POSITION_DATA non-empty; renderer suppressing display.")

                
            # label and indicator
            try:
                y_off = 18 if label == "LEFT" else 36
                right_surf.blit(small_font.render(f"{label}: {shown_type}", True, COLOR_TEXT), (6, y_off))
                ind_color = INDICATOR_REAL if frame_type == "REAL" else INDICATOR_GHOST
                pygame.draw.circle(right_surf, ind_color, (WIDTH - 18, y_off + 6), 6)
            except Exception:
                pass

        # --- NOW create panels after all drawing is done ---
        left_panel = pygame.Surface((WIDTH + BORDER * 2, HEIGHT + BORDER * 2))
        left_panel.fill(COLOR_BORDER)
        left_panel.blit(left_surf, (BORDER, BORDER))

        right_panel = pygame.Surface((WIDTH + BORDER * 2, HEIGHT + BORDER * 2))
        right_panel.fill(COLOR_BORDER)
        right_panel.blit(right_surf, (BORDER, BORDER))

        # Diagnostics overlay (frame counter, timestamp)
        try:
            now = datetime.now().strftime("%H:%M:%S")
            diag_text = f"Frame: {frame_idx}  Time: {now}"
            screen.fill(COLOR_BG)
            # blit panels onto screen with spacing
            screen.blit(left_panel, (BORDER, BORDER))
            screen.blit(right_panel, (WIDTH + BORDER * 2, BORDER))
            # titles
            screen.blit(title_font.render("AR (with generation) - current ar_data", True, COLOR_TITLE), (BORDER + 6, 6))
            screen.blit(title_font.render("AR (without generation) - real-only", True, COLOR_TITLE), (WIDTH + BORDER * 2 + 6, 6))
            screen.blit(diag_font.render(diag_text, True, COLOR_DIAG), (6, HEIGHT + BORDER + 6))
        except Exception:
            screen.blit(left_panel, (BORDER, BORDER))
            screen.blit(right_panel, (WIDTH + BORDER * 2, BORDER))

        # Final flip then capture to ensure captured frame matches display
        pygame.display.flip()
        clock.tick(FPS)

        # capture combined frame after flip
        frames.append(surf_to_frame(screen, scale=SCALE))

        # handle events (allow early quit)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

    # cleanup
    cap.release()
    pygame.quit()

    # If we captured frames, drop the very last frame and save looping GIF
    if frames:
        if len(frames) > 1:
            frames_to_save = frames[:-1]  # drop last frame
        else:
            frames_to_save = frames
        try:
            imageio.mimsave(OUT_GIF, frames_to_save, fps=FPS, loop=0)
            print("Saved", OUT_GIF)
        except Exception as e:
            print("Failed to save GIF:", e)
    else:
        print("No frames captured; GIF not saved.")


if __name__ == "__main__":
    main()
