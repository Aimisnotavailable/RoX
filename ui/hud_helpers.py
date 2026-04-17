import pygame as pg
import math

RADIAL_MENU_RADIUS = 160

def draw_rounded_rect(surface, rect, color, radius=8, border_width=0, border_color=None):
    if rect.width <= 0 or rect.height <= 0:
        return
    temp = pg.Surface((rect.width, rect.height), pg.SRCALPHA)
    pg.draw.rect(temp, color, (0, 0, rect.width, rect.height), border_radius=radius)
    if border_width > 0 and border_color:
        pg.draw.rect(temp, border_color, (0, 0, rect.width, rect.height), border_width, border_radius=radius)
    surface.blit(temp, rect.topleft)

def draw_glow_rect(surface, rect, color, radius=8, glow_size=4):
    rgb = color[:3] if len(color) == 4 else color
    for i in range(glow_size, 0, -1):
        alpha = int(30 * (1 - i/glow_size))
        glow_color = (*rgb, alpha)
        glow_rect = rect.inflate(i*2, i*2)
        draw_rounded_rect(surface, glow_rect, glow_color, radius=radius+i)

class RadialMenu:
    def __init__(self):
        self.active = False
        self.center = (0, 0)
        self.selected_index = -1
        self.menu_stack = []
        self.current_options = None
        self.font = pg.font.SysFont('Consolas', 16, bold=True)
        self.pulse = 0.0

    def activate(self, center, top_level_options):
        self.active = True
        self.center = center
        self.menu_stack = [top_level_options]
        self.current_options = top_level_options
        self.selected_index = -1
        self.pulse = 0.0

    def deactivate(self):
        self.active = False
        self.menu_stack = []
        self.current_options = None

    def push_submenu(self, options):
        self.menu_stack.append(options)
        self.current_options = options
        self.selected_index = -1

    def pop_submenu(self):
        if len(self.menu_stack) > 1:
            self.menu_stack.pop()
            self.current_options = self.menu_stack[-1]
            self.selected_index = -1

    def update_selection(self, screen_point):
        if not self.active:
            return
        dx = screen_point[0] - self.center[0]
        dy = screen_point[1] - self.center[1]
        dist = math.hypot(dx, dy)
        if dist < 40:
            self.selected_index = -1
            return
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        n = len(self.current_options)
        sector = int(angle / (2 * math.pi / n))
        self.selected_index = sector % n

    def draw(self, surface, pulse_factor):
        if not self.active:
            return
        cx, cy = self.center
        options = self.current_options
        n = len(options)

        # Outer glow (pulsing)
        glow_alpha = int(40 + 20 * math.sin(pulse_factor * 8))
        for r in range(RADIAL_MENU_RADIUS + 4, RADIAL_MENU_RADIUS + 12, 2):
            alpha = glow_alpha * (1 - (r - (RADIAL_MENU_RADIUS+4))/8)
            pg.draw.circle(surface, (100, 150, 255, int(alpha)), (cx, cy), r, 2)

        # Dark glass base
        bg_surf = pg.Surface((RADIAL_MENU_RADIUS*2, RADIAL_MENU_RADIUS*2), pg.SRCALPHA)
        pg.draw.circle(bg_surf, (10, 15, 25, 220), (RADIAL_MENU_RADIUS, RADIAL_MENU_RADIUS), RADIAL_MENU_RADIUS)
        surface.blit(bg_surf, (cx - RADIAL_MENU_RADIUS, cy - RADIAL_MENU_RADIUS))

        # Draw sectors
        for i, option in enumerate(options):
            start_angle = i * (2 * math.pi / n)
            end_angle = (i + 1) * (2 * math.pi / n)
            # Color gradient from base color to a lighter version
            base_color = option["color"]
            if i == self.selected_index:
                # Highlight
                color = (255, 255, 100)
                # Add a glow around the sector
                points = []
                for t in range(0, 11):
                    angle = start_angle + (end_angle - start_angle) * (t / 10)
                    x = cx + math.cos(angle) * (RADIAL_MENU_RADIUS + 4)
                    y = cy + math.sin(angle) * (RADIAL_MENU_RADIUS + 4)
                    points.append((x, y))
                pg.draw.polygon(surface, (*color, 100), points)
            else:
                color = base_color

            # Draw sector polygon
            points = [(cx, cy)]
            for t in range(0, 11):
                angle = start_angle + (end_angle - start_angle) * (t / 10)
                x = cx + math.cos(angle) * RADIAL_MENU_RADIUS
                y = cy + math.sin(angle) * RADIAL_MENU_RADIUS
                points.append((x, y))
            pg.draw.polygon(surface, (*color, 180), points)

            # Dividing lines (inner and outer)
            line_x = cx + math.cos(start_angle) * RADIAL_MENU_RADIUS
            line_y = cy + math.sin(start_angle) * RADIAL_MENU_RADIUS
            pg.draw.line(surface, (150, 150, 200, 200), (cx, cy), (line_x, line_y), 2)
            # Outer rim
            line_x_outer = cx + math.cos(start_angle) * (RADIAL_MENU_RADIUS + 2)
            line_y_outer = cy + math.sin(start_angle) * (RADIAL_MENU_RADIUS + 2)
            pg.draw.line(surface, (150, 150, 200, 200), (cx, cy), (line_x_outer, line_y_outer), 1)

        # Last dividing line (full circle)
        last_angle = 2 * math.pi
        line_x = cx + math.cos(last_angle) * RADIAL_MENU_RADIUS
        line_y = cy + math.sin(last_angle) * RADIAL_MENU_RADIUS
        pg.draw.line(surface, (150, 150, 200, 200), (cx, cy), (line_x, line_y), 2)

        # Labels with shadow
        for i, option in enumerate(options):
            mid_angle = (i + 0.5) * (2 * math.pi / n)
            label_x = cx + math.cos(mid_angle) * (RADIAL_MENU_RADIUS * 0.7)
            label_y = cy + math.sin(mid_angle) * (RADIAL_MENU_RADIUS * 0.7)
            # Render text with shadow
            text_surf = self.font.render(option["name"], True, (255, 255, 255))
            shadow_surf = self.font.render(option["name"], True, (0, 0, 0))
            text_rect = text_surf.get_rect(center=(label_x, label_y))
            shadow_rect = text_rect.copy()
            shadow_rect.x += 2
            shadow_rect.y += 2
            surface.blit(shadow_surf, shadow_rect)
            surface.blit(text_surf, text_rect)

        # Inner ring
        pg.draw.circle(surface, (100, 150, 200, 150), (cx, cy), 18, 3)
        pg.draw.circle(surface, (255, 255, 255, 80), (cx, cy), 12, 1)