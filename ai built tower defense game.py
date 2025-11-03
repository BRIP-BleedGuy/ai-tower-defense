# tower_defense_dev_final_fixed_v3.py
# Full game file with:
# - forced-turn paths
# - sandbox (.bri) and full game (.BRIP) saves into user save folder
# - pause menu with semi-transparent overlay
# - dev code "uuuu" via hidden top-right input, Dev Menu, and moved dev button
# - Dev Menu spawn now spawns one enemy immediately and spawn_enemy ensures on-screen starts
# - Progressive scaling: enemies get stronger and faster after round 10
import pygame, math, random, sys, time, json, os

# Optional numpy for tones (safe if missing)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

pygame.init()
AUDIO_AVAILABLE = True
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=1)
except Exception:
    AUDIO_AVAILABLE = False

def make_tone_sound(freq=440, duration_ms=150, volume=0.5):
    if not AUDIO_AVAILABLE or not NUMPY_AVAILABLE:
        return None
    try:
        sr = 44100
        length = int(sr * (duration_ms / 1000.0))
        t = np.linspace(0, duration_ms / 1000.0, length, False)
        wave = 0.5 * np.sin(2 * math.pi * freq * t)
        envelope = np.linspace(1.0, 0.001, wave.shape[0])
        wave = wave * envelope
        audio = np.int16(wave * 32767 * volume)
        return pygame.sndarray.make_sound(audio)
    except Exception:
        return None

def play_sound(sound):
    if not AUDIO_AVAILABLE or sound is None:
        return
    try:
        sound.play()
    except Exception:
        pass

# basic tones
SOUND_HIT = make_tone_sound(880, 100, 0.25)
SOUND_SPLASH = make_tone_sound(520, 200, 0.28)
SOUND_STUN = make_tone_sound(220, 220, 0.35)
SOUND_LASER = make_tone_sound(1200, 60, 0.08)
SOUND_MORTAR = make_tone_sound(300, 260, 0.22)

# ---- Constants ----
WIDTH, HEIGHT = 1000, 700
GRID_SIZE = 40
MAP_W, MAP_H = 18, 13
PLAY_W, PLAY_H = MAP_W * GRID_SIZE, MAP_H * GRID_SIZE

WHITE=(255,255,255); BLACK=(10,10,10); LIGHT_GRAY=(200,200,200); GRAY=(130,130,130)
RED=(220,60,60); GREEN=(60,200,80); BLUE=(80,150,240); PURPLE=(160,60,200); BROWN=(120,70,20)

FPS = 60
FONT = pygame.font.SysFont("Arial", 18)
BIG = pygame.font.SysFont("Arial", 32)
SMALL = pygame.font.SysFont("Arial", 14)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tower Defense — Dev Tools (v3)")

def draw_text(surf, text, pos, color=BLACK, font=FONT):
    surf.blit(font.render(str(text), True, color), pos)

def draw_centered_text(surf, text, rect, color=BLACK, font=FONT):
    txt = font.render(str(text), True, color)
    surf.blit(txt, (rect.x + (rect.width - txt.get_width())//2, rect.y + (rect.height - txt.get_height())//2))

def distance(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

def cell_to_pixel(cell):
    x,y = cell
    return x*GRID_SIZE + GRID_SIZE//2, y*GRID_SIZE + GRID_SIZE//2

def pixel_to_cell(px,py):
    if px<0 or px>=PLAY_W or py<0 or py>=PLAY_H: return None
    return px//GRID_SIZE, py//GRID_SIZE

# ---- Enemy / Tower classes ----
class Enemy:
    def __init__(self, path_pixels, kind="normal"):
        self.kind = kind
        self.path = path_pixels if path_pixels else [(0, PLAY_H//2)]
        self.x, self.y = self.path[0]
        self.index = 0
        self.effects = {"slow": [], "stun": []}
        if kind=="normal":
            self.max_health=100; self.speed=1.0; self.reward=10; self.color=RED; self.base_damage=1; self.name="Grunt"
        elif kind=="fast":
            self.max_health=70; self.speed=1.6; self.reward=12; self.color=BLUE; self.base_damage=1; self.name="Runner"
        elif kind=="heavy":
            self.max_health=300; self.speed=0.6; self.reward=40; self.color=BROWN; self.base_damage=5; self.name="Brute"
        elif kind=="boss":
            self.max_health=1200; self.speed=0.7; self.reward=200; self.color=PURPLE; self.base_damage=20; self.name="Boss"
        elif kind=="shield":
            self.max_health=180; self.speed=0.9; self.reward=25; self.color=(240,220,60); self.armor=0.5; self.base_damage=2; self.name="Shield"
        elif kind=="stunner":
            self.max_health=90; self.speed=1.0; self.reward=30; self.color=(180,80,80); self.base_damage=2; self.name="Stunner"
            self.stun_radius=80; self.stun_chance=0.02; self.stun_duration=90
        else:
            self.max_health=100; self.speed=1.0; self.reward=10; self.color=RED; self.base_damage=1; self.name=kind
        self.health=self.max_health; self.radius=12; self.base_speed=self.speed

    def apply_effects(self):
        total_slow = sum(a for (a,r) in self.effects["slow"])
        eff = self.base_speed * max(0.05, 1 - total_slow)
        self.effects["slow"] = [(a, r-1) for (a,r) in self.effects["slow"] if r-1>0]
        stun_active = any(r>0 for (_,r) in self.effects["stun"])
        self.effects["stun"] = [(a, r-1) for (a,r) in self.effects["stun"] if r-1>0]
        if stun_active: return 0.0
        return eff

    def update(self):
        if self.index + 1 >= len(self.path): return False
        eff = self.apply_effects()
        tx,ty = self.path[self.index+1]
        dx,dy = tx-self.x, ty-self.y
        dist = math.hypot(dx,dy)
        if dist < eff:
            self.x, self.y = tx, ty
            self.index += 1
        else:
            if eff>0:
                self.x += eff*dx/dist; self.y += eff*dy/dist
        return True

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), self.radius)
        w,h = 36,6; hx=int(self.x - w/2); hy=int(self.y - self.radius - 10)
        pygame.draw.rect(surf, LIGHT_GRAY, (hx,hy,w,h))
        pygame.draw.rect(surf, GREEN, (hx,hy,max(0, w*(self.health/self.max_health)), h))
        pygame.draw.rect(surf, BLACK, (hx,hy,w,h),1)

    def take_damage(self, dmg):
        if hasattr(self,"armor"): dmg *= (1 - self.armor)
        self.health -= dmg

    def add_slow(self, amount, frames): self.effects["slow"].append((amount, frames))
    def add_stun(self, frames): self.effects["stun"].append((1, frames))

class Tower:
    def __init__(self, grid_pos, kind="basic"):
        self.grid_pos = tuple(grid_pos)
        self.x, self.y = grid_pos[0]*GRID_SIZE + GRID_SIZE//2, grid_pos[1]*GRID_SIZE + GRID_SIZE//2
        self.kind = kind; self.level = 1
        self.set_base_stats()
        self.cooldown_timer = 0; self.stunned_for = 0

    def set_base_stats(self):
        if self.kind=="basic":
            self.range=120; self.damage=25; self.fire_rate=50; self.cost=50; self.color=GRAY; self.splash=0
        elif self.kind=="sniper":
            self.range=260; self.damage=75; self.fire_rate=120; self.cost=120; self.color=BLUE; self.splash=0
        elif self.kind=="splash":
            self.range=110; self.damage=20; self.fire_rate=90; self.cost=90; self.color=PURPLE; self.splash=70
        elif self.kind=="freeze":
            self.range=140; self.damage=8; self.fire_rate=100; self.cost=80; self.color=LIGHT_GRAY; self.splash=0
            self.slow_amount=0.45; self.slow_time=120
        elif self.kind=="laser":
            self.range=100; self.damage=8; self.fire_rate=6; self.cost=140; self.color=(255,100,100); self.splash=0
        elif self.kind=="mortar":
            self.range=220; self.damage=60; self.fire_rate=160; self.cost=150; self.color=(140,100,40); self.splash=90
        else:
            self.range=100; self.damage=20; self.fire_rate=60; self.cost=50; self.color=GRAY; self.splash=0
        self.upgrade_cost = int(self.cost * 0.8)

    def is_stunned(self): return self.stunned_for > 0
    def apply_stun(self, frames): self.stunned_for = max(self.stunned_for, frames)
    def can_shoot(self): return self.cooldown_timer <= 0 and not self.is_stunned()
    def update_timer(self):
        if self.cooldown_timer>0: self.cooldown_timer -= 1
        if self.stunned_for>0: self.stunned_for -= 1

    def shoot(self, enemies, sound_on=True):
        self.update_timer()
        if not self.can_shoot(): return []
        targets = [e for e in enemies if distance((self.x,self.y),(e.x,e.y)) <= self.range]
        if not targets: return []
        targets.sort(key=lambda e: (e.index, -distance((self.x,self.y),(e.x,e.y))), reverse=True)
        primary = targets[0]; projs=[]
        if self.kind=="sniper":
            primary.take_damage(self.damage); self.cooldown_timer=self.fire_rate
            if sound_on: play_sound(SOUND_HIT); projs.append(("hit", primary.x, primary.y)); return projs
        elif self.kind=="freeze":
            primary.take_damage(self.damage); primary.add_slow(self.slow_amount, self.slow_time); self.cooldown_timer=self.fire_rate
            if sound_on: play_sound(SOUND_HIT); projs.append(("freeze", primary.x, primary.y)); return projs
        elif self.kind=="splash":
            for e in enemies:
                if distance((primary.x,primary.y),(e.x,e.y)) <= self.splash: e.take_damage(self.damage)
            self.cooldown_timer=self.fire_rate
            if sound_on: play_sound(SOUND_SPLASH); projs.append(("splash", primary.x, primary.y)); return projs
        elif self.kind=="laser":
            primary.take_damage(self.damage); self.cooldown_timer = self.fire_rate
            if sound_on: play_sound(SOUND_LASER); projs.append(("laser", primary.x, primary.y)); return projs
        elif self.kind=="mortar":
            for e in enemies:
                if distance((primary.x,primary.y),(e.x,e.y)) <= self.splash: e.take_damage(self.damage)
            self.cooldown_timer = self.fire_rate
            if sound_on: play_sound(SOUND_MORTAR); projs.append(("mortar", primary.x, primary.y)); return projs
        else:
            primary.take_damage(self.damage); self.cooldown_timer=self.fire_rate
            if sound_on: play_sound(SOUND_HIT); projs.append(("hit", primary.x, primary.y)); return projs

    def upgrade(self):
        self.level += 1
        self.damage = int(self.damage * 1.35)
        self.range = int(self.range * 1.12)
        self.fire_rate = max(6, int(self.fire_rate * 0.85))
        self.upgrade_cost = int(self.upgrade_cost * 1.9)

    def draw(self, surf, selected=False):
        r = 16 + self.level
        pygame.draw.circle(surf, self.color, (self.x, self.y), r)
        pygame.draw.circle(surf, BLACK, (self.x, self.y), r, 2)
        if selected: pygame.draw.circle(surf, (0,0,0), (self.x,self.y), self.range, 1)
        if self.is_stunned(): pygame.draw.circle(surf, (180,180,255), (self.x,self.y), r-4, 2)
        max_cd = max(1, self.fire_rate)
        bx = self.x - 14; by = self.y - r - 12; bar_w, bar_h = 28, 6
        if self.cooldown_timer > 0:
            prop = self.cooldown_timer / max_cd
            pygame.draw.rect(surf, LIGHT_GRAY, (bx,by,bar_w,bar_h))
            pygame.draw.rect(surf, RED, (bx,by,int(bar_w*prop),bar_h))
            pygame.draw.rect(surf, BLACK, (bx,by,bar_w,bar_h),1)
        else:
            pygame.draw.rect(surf, GREEN, (bx,by,bar_w,bar_h)); pygame.draw.rect(surf, BLACK, (bx,by,bar_w,bar_h),1)

# ---- Game manager (with ESC -> Pause and hidden dev input) ----
class Game:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.reset_state()
        self.create_shop_buttons()
        self.menu_state = "main"  # main / settings / sandbox_draw / playing / pause_menu
        self.exit_confirm_time = 0.0; self.exit_confirm_window = 1.5
        self.fullscreen = False; self.sound_on = True; self.auto_start = False
        self.io_message = "" ; self.io_message_time = 0.0
        # dev flag and UI
        self.dev_mode_unlocked = False
        self.show_dev_button = False
        # moved dev button lower (y increased)
        self.dev_button_rect = pygame.Rect(8, 520, 120, 32)
        self.forced_spawn_type = None
        # hidden dev-code input (top-right area)
        self.dev_code_buffer = ""
        self.dev_code_focused = False
        self.dev_code_rect = pygame.Rect(WIDTH - 150, 10, 140, 28)
        # safe save folder in user's home to avoid permission issues
        self.save_dir = os.path.join(os.path.expanduser("~"), "TowerDefenseSaves")
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            # fall back to current directory if creation fails
            self.save_dir = os.getcwd()

    def reset_state(self):
        self.enemies = []; self.towers = []; self.projectile_effects = []
        self.money = 250; self.base_health = 50; self.wave = 1
        self.wave_in_progress = False; self.spawn_timer = 0; self.to_spawn = 0
        self.selected_tower_type = "basic"; self.selected_tower = None
        self.level_info = "Press SPACE to start wave"
        self.paused = False; self.game_over = False
        self.forbidden = set()
        self.path_cells = []; self.path_pixels = []
        self.mode = None

    def create_shop_buttons(self):
        x0 = PLAY_W + 10; y0 = 160
        choices = [("basic","Basic",50), ("sniper","Sniper",120), ("splash","Splash",90), ("freeze","Freeze",80),
                   ("laser","Laser",140), ("mortar","Mortar",150)]
        self.shop_buttons = []
        y = y0
        for kind,label,cost in choices:
            rect = pygame.Rect(x0, y, 150, 36)
            self.shop_buttons.append((kind, rect, cost, label))
            y += 48

    # ---- forced-turn path generator (keeps earlier rules) ----
    def gen_path_turning(self, winding_factor=0.3, force_turn_window=3):
        center = MAP_H // 2
        center_rows = [r for r in (center-1, center, center+1) if 0 <= r < MAP_H]
        x = 0; y = random.choice(center_rows)
        cells = [(x, y)]; turned = False
        while x < MAP_W - 1:
            choices = [(x+1, y)]
            vertical_bias = winding_factor
            if not turned and x < force_turn_window:
                vertical_bias = min(1.0, winding_factor + 0.6)
            if random.random() < vertical_bias:
                if y > 0: choices.append((x, y-1))
                if y < MAP_H - 1: choices.append((x, y+1))
            if random.random() < winding_factor * 0.6:
                if y > 0: choices.append((x+1, y-1))
                if y < MAP_H - 1: choices.append((x+1, y+1))
            weights = [0.6 if nx > x else 0.4 for (nx,ny) in choices]
            nx, ny = random.choices(choices, weights=weights, k=1)[0]
            if nx == x and ny != y: turned = True
            elif nx == x+1 and ny != y: turned = True
            if (nx, ny) != cells[-1]:
                cells.append((nx, ny)); x, y = nx, ny
            if (not turned) and (x >= force_turn_window - 1):
                ly = cells[-1][1]
                if ly > 0 and (cells[-1][0], ly - 1) not in cells:
                    cells.append((cells[-1][0], ly - 1)); y = ly - 1; turned = True
                elif ly < MAP_H - 1 and (cells[-1][0], ly + 1) not in cells:
                    cells.append((cells[-1][0], ly + 1)); y = ly + 1; turned = True
                x = cells[-1][0]
        cleaned = []
        for c in cells:
            if not cleaned or c != cleaned[-1]:
                cleaned.append(c)
        lastx, lasty = cleaned[-1]
        while lastx < MAP_W - 1:
            lastx += 1; cleaned.append((lastx, lasty))
        if not any((cleaned[i+1][0] == cleaned[i][0] and cleaned[i+1][1] != cleaned[i][1]) or
                   (cleaned[i+1][0] == cleaned[i][0] + 1 and cleaned[i+1][1] != cleaned[i][1])
                   for i in range(len(cleaned)-1)):
            mid = max(1, MAP_W // 3)
            mx, my = cleaned[mid]
            if my > 0: cleaned.insert(mid+1, (mx, my-1))
            else: cleaned.insert(mid+1, (mx, min(MAP_H-1, my+1)))
        self.path_cells = cleaned
        self.path_pixels = [cell_to_pixel(c) for c in self.path_cells]
        self.forbidden = set(self.path_cells)

    def gen_easy(self): self.gen_path_turning(winding_factor=0.18, force_turn_window=3)
    def gen_hard(self): self.gen_path_turning(winding_factor=0.6, force_turn_window=3)

    # ---- sandbox save/load in .bri ----
    def start_sandbox(self):
        self.mode = "sandbox"
        self.path_cells = []; self.path_pixels = []; self.forbidden = set()
        self.level_info = "Draw path: start on left edge, end on right edge. Click to add, right-click to undo. Enter to play."

    def sandbox_add_cell(self, cell):
        if cell is None: return
        if not (0 <= cell[0] < MAP_W and 0 <= cell[1] < MAP_H): return
        if not self.path_cells:
            if cell[0] != 0:
                self.level_info = "First cell must be on left edge."; return
            self.path_cells.append(cell); self.level_info = "Added start; continue."; return
        lx,ly = self.path_cells[-1]; nx,ny = cell
        if abs(nx - lx) + abs(ny - ly) != 1:
            self.level_info = "Cell must be 4-way adjacent."; return
        if cell in self.path_cells:
            self.level_info = "Cell already in path."; return
        self.path_cells.append(cell)
        if cell[0] == MAP_W - 1:
            self.level_info = "Path reaches right edge. Press Enter to play."
        else:
            self.level_info = f"Added {cell}. Continue."

    def sandbox_remove_last(self):
        if self.path_cells:
            removed = self.path_cells.pop(); self.level_info = f"Removed {removed}."
        else:
            self.level_info = "Nothing to remove."

    def sandbox_finalize(self):
        if not self.path_cells:
            self.level_info = "No path drawn."; return False
        if self.path_cells[0][0] != 0:
            self.level_info = "Path must start on left edge."; return False
        if self.path_cells[-1][0] != MAP_W - 1:
            self.level_info = "Path must end on right edge."; return False
        self.path_pixels = [cell_to_pixel(c) for c in self.path_cells]
        self.forbidden = set(self.path_cells); self.level_info = "Sandbox path valid."; return True

    def save_sandbox(self, filename):
        if not filename.lower().endswith(".bri"):
            filename += ".bri"
        path = filename if os.path.isabs(filename) else os.path.join(self.save_dir, filename)
        try:
            data = {"path_cells": self.path_cells}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            self.io_message = f"Sandbox saved to {os.path.basename(path)} in {self.save_dir}"
            self.io_message_time = time.time()
            return True
        except Exception as ex:
            self.io_message = f"Sandbox save failed: {ex}"
            self.io_message_time = time.time()
            return False

    def load_sandbox(self, filename):
        if not filename.lower().endswith(".bri"):
            self.io_message = "Sandbox load must be .bri"; self.io_message_time = time.time(); return False
        path = filename if os.path.isabs(filename) else os.path.join(self.save_dir, filename)
        if not os.path.exists(path):
            self.io_message = "Sandbox file not found"; self.io_message_time = time.time(); return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cells = [tuple(c) for c in data.get("path_cells", [])]
            if not cells:
                self.io_message = "Invalid sandbox file"; self.io_message_time = time.time(); return False
            self.path_cells = cells; self.path_pixels = [cell_to_pixel(c) for c in self.path_cells]; self.forbidden = set(self.path_cells)
            self.io_message = f"Sandbox loaded {os.path.basename(path)}"
            self.io_message_time = time.time(); return True
        except Exception as ex:
            self.io_message = f"Sandbox load failed: {ex}"; self.io_message_time = time.time(); return False

    # ---- save/load full game .BRIP (uses safe save dir) ----
    def save_game(self, filename):
        if not filename.lower().endswith(".brip"):
            filename += ".BRIP"
        path = filename if os.path.isabs(filename) else os.path.join(self.save_dir, filename)
        try:
            data = {
                "money": self.money,
                "base_health": self.base_health,
                "wave": self.wave,
                "wave_in_progress": self.wave_in_progress,
                "to_spawn": self.to_spawn,
                "spawn_timer": self.spawn_timer,
                "mode": self.mode,
                "path_cells": self.path_cells,
                "towers": [
                    {"grid_pos": list(t.grid_pos), "kind": t.kind, "level": t.level} for t in self.towers
                ],
                "enemies": [
                    {"index": e.index, "kind": e.kind, "health": e.health} for e in self.enemies
                ],
                "settings": {
                    "auto_start": self.auto_start,
                    "fullscreen": self.fullscreen,
                    "sound_on": self.sound_on
                }
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            self.io_message = f"Saved to {os.path.basename(path)} in {self.save_dir}"
            self.io_message_time = time.time(); return True
        except Exception as ex:
            self.io_message = f"Save failed: {ex}"
            self.io_message_time = time.time(); return False

    def load_game(self, filename):
        if not filename.lower().endswith(".brip"):
            self.io_message = "Load failed: file must end with .BRIP"; self.io_message_time = time.time(); return False
        path = filename if os.path.isabs(filename) else os.path.join(self.save_dir, filename)
        if not os.path.exists(path):
            self.io_message = "Load failed: file not found"; self.io_message_time = time.time(); return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            required_keys = {"money","base_health","wave","path_cells","towers","settings"}
            if not required_keys.issubset(set(data.keys())):
                self.io_message = "Load failed: invalid file"; self.io_message_time = time.time(); return False
            self.money = int(data.get("money", 250))
            self.base_health = int(data.get("base_health", 50))
            self.wave = int(data.get("wave", 1))
            self.wave_in_progress = bool(data.get("wave_in_progress", False))
            self.to_spawn = int(data.get("to_spawn", 0))
            self.spawn_timer = int(data.get("spawn_timer", 0))
            self.mode = data.get("mode", None)
            self.path_cells = [tuple(c) for c in data.get("path_cells", [])]
            self.path_pixels = [cell_to_pixel(c) for c in self.path_cells]
            self.forbidden = set(self.path_cells)
            self.towers = []
            for td in data.get("towers", []):
                gp = tuple(td.get("grid_pos", (0,0)))
                kind = td.get("kind", "basic")
                t = Tower(gp, kind)
                t.level = int(td.get("level", 1))
                for _ in range(1, t.level):
                    t.upgrade()
                self.towers.append(t)
            self.enemies = []
            for ed in data.get("enemies", []):
                kind = ed.get("kind", "normal")
                e = Enemy(self.path_pixels, kind)
                e.index = int(ed.get("index", 0))
                e.health = float(ed.get("health", e.max_health))
                self.enemies.append(e)
            settings = data.get("settings", {})
            self.auto_start = bool(settings.get("auto_start", False))
            self.fullscreen = bool(settings.get("fullscreen", False))
            self.sound_on = bool(settings.get("sound_on", True))
            self.menu_state = "playing"
            self.io_message = f"Loaded {os.path.basename(path)}"
            self.io_message_time = time.time(); return True
        except Exception as ex:
            self.io_message = f"Load failed: {ex}"
            self.io_message_time = time.time(); return False

    # ---- spawn/update; stunners spawn only after wave > 5 ----
    def start_wave(self):
        if self.wave_in_progress or self.game_over: return
        if self.wave % 7 == 0:
            self.to_spawn = 1 + self.wave // 7; self.enemy_kind = "boss"
        else:
            self.to_spawn = 5 + self.wave * 1; self.enemy_kind = "mixed"
        self.spawn_timer = 0; self.wave_in_progress = True; self.level_info = f"Wave {self.wave} started"

    def spawn_enemy(self, kind=None, consume_to_spawn=False):
        """
        Spawn an enemy using the current path_pixels.
        - kind: optional string to force enemy type.
        - consume_to_spawn: if True this call decrements self.to_spawn (used by normal waves).
        Ensures on-screen spawn and applies progressive scaling after round 10.
        """
        # Ensure valid path_pixels (fallback straight middle row)
        if not self.path_pixels:
            mid_row = MAP_H // 2
            self.path_cells = [(x, mid_row) for x in range(MAP_W)]
            self.path_pixels = [cell_to_pixel(c) for c in self.path_cells]
            self.forbidden = set(self.path_cells)

        # Decide kind if not provided
        if kind is None:
            if getattr(self, "enemy_kind", None) == "boss":
                kind = "boss"
            elif hasattr(self, "forced_spawn_type") and self.forced_spawn_type:
                kind = self.forced_spawn_type
                if consume_to_spawn:
                    # clear forced spawn only when used in normal wave consumption
                    self.forced_spawn_type = None
            else:
                r = random.random()
                if self.wave > 5:
                    if r < 0.55: kind = "normal"
                    elif r < 0.75: kind = "fast"
                    elif r < 0.9: kind = "stunner"
                    else: kind = "heavy"
                else:
                    if r < 0.65: kind = "normal"
                    elif r < 0.9: kind = "fast"
                    else: kind = "heavy"

        # Create enemy with current path
        e = Enemy(list(self.path_pixels), kind)
        # Place it exactly at the start of the path so it is visible
        e.x, e.y = self.path_pixels[0]
        e.index = 0

        # Progressive scaling after round 10
        if self.wave > 10:
            over = max(0, self.wave - 10)
            # base multipliers per round beyond 10
            if kind == "boss":
                health_per_round = 0.18   # +18% per round for bosses
                speed_per_round = 0.05    # +5% per round for bosses
                max_health_mult = 8.0
                max_speed_mult = 2.5
            else:
                health_per_round = 0.12   # +12% per round for regular enemies
                speed_per_round = 0.03    # +3% per round for regular enemies
                max_health_mult = 5.0
                max_speed_mult = 2.0

            # compute multipliers (linear growth)
            health_mult = 1.0 + health_per_round * over
            speed_mult = 1.0 + speed_per_round * over

            # cap them
            health_mult = min(health_mult, max_health_mult)
            speed_mult = min(speed_mult, max_speed_mult)

            # apply to the enemy
            e.max_health = float(e.max_health) * health_mult
            # keep current health proportional to new max
            e.health = float(e.health) * health_mult
            # scale base and current speed
            e.base_speed = float(e.base_speed) * speed_mult
            # some enemies reference 'speed' directly
            e.speed = getattr(e, "speed", e.base_speed) * speed_mult

            # scale reward modestly with health (so stronger enemies pay more)
            e.reward = int(getattr(e, "reward", 0) * health_mult)

            # for shield enemies, keep armor unchanged
            if hasattr(e, "armor"):
                pass

        # Add to the list
        self.enemies.append(e)

        # Decrement to_spawn if this was intended for a wave spawn
        if consume_to_spawn:
            self.to_spawn = max(0, self.to_spawn - 1)

        return e

    def update(self):
        if self.menu_state != "playing": return
        if self.paused or self.game_over: return
        dt = self.clock.tick(FPS)
        if self.wave_in_progress:
            self.spawn_timer += 1
            if self.spawn_timer >= 40 and self.to_spawn > 0:
                # spawn for wave and decrement to_spawn
                if self.enemy_kind == "boss":
                    self.spawn_enemy(kind="boss", consume_to_spawn=True)
                else:
                    # if forced spawn type exists and is intended for wave, use and clear it
                    if hasattr(self, "forced_spawn_type") and self.forced_spawn_type:
                        self.spawn_enemy(kind=self.forced_spawn_type, consume_to_spawn=True)
                        self.forced_spawn_type = None
                    else:
                        # call spawn_enemy without specifying kind so probabilities apply
                        spawned = self.spawn_enemy(consume_to_spawn=True)
                self.spawn_timer = 0
            if self.to_spawn <= 0 and not self.enemies:
                self.wave_in_progress = False; self.wave += 1; self.money += 50 + self.wave * 5
                self.level_info = f"Wave {self.wave-1} cleared. Press SPACE or Auto-start."
                if self.auto_start: self.start_wave()
        for e in self.enemies[:]:
            alive = e.update()
            if e.kind == "stunner" and random.random() < e.stun_chance:
                for t in self.towers:
                    if distance((e.x,e.y),(t.x,t.y)) <= e.stun_radius:
                        t.apply_stun(e.stun_duration)
                self.projectile_effects.append(["stun_pulse", e.x, e.y, 18])
                if self.sound_on: play_sound(SOUND_STUN)
            if not alive:
                self.base_health -= e.base_damage; self.enemies.remove(e)
                if self.base_health <= 0: self.game_over = True
            elif e.health <= 0:
                self.money += e.reward; self.enemies.remove(e)
        for t in self.towers:
            projs = t.shoot(self.enemies, sound_on=self.sound_on)
            for p in projs:
                typ, px, py = p[0], p[1], p[2]; self.projectile_effects.append([typ, px, py, 18])
        for p in self.projectile_effects[:]:
            p[3] -= 1
            if p[3] <= 0: self.projectile_effects.remove(p)

    # ---- placement ----
    def can_place_at(self, cell):
        if cell is None: return False
        gx,gy = cell
        if not (0 <= gx < MAP_W and 0 <= gy < MAP_H): return False
        if cell in self.forbidden: return False
        for t in self.towers:
            if t.grid_pos == cell: return False
        return True

    def place_tower(self, cell):
        if not self.can_place_at(cell):
            self.level_info = "Cannot place there."; return False
        cost = Tower(cell, self.selected_tower_type).cost
        if self.money < cost:
            self.level_info = "Not enough money!"; return False
        self.money -= cost; self.towers.append(Tower(cell, self.selected_tower_type))
        self.level_info = f"Placed {self.selected_tower_type}"; return True

    def sell_tower(self, tower):
        if tower in self.towers:
            sell_value = int(tower.cost * (0.5 + 0.1 * tower.level)); self.money += sell_value; self.towers.remove(tower); self.selected_tower = None

    def upgrade_tower(self, tower):
        if tower is None: return
        if self.money < tower.upgrade_cost:
            self.level_info = "Not enough money to upgrade!"; return
        self.money -= tower.upgrade_cost; tower.upgrade()

    # ---- Dev code unlock and menu (modified dev code = "uuuu") ----
    def enter_dev_code_from_buffer(self, code):
        if code and code.strip().lower() == "uuuu":
            self.dev_mode_unlocked = True
            self.money += 500000
            self.show_dev_button = True
            self.io_message = "Dev code accepted: +$500000. Dev menu enabled."
            self.io_message_time = time.time()
        else:
            self.io_message = "Invalid dev code"
            self.io_message_time = time.time()

    def enter_dev_code(self):
        code = self.prompt_text("Enter dev code:")
        if code:
            self.enter_dev_code_from_buffer(code)

    # helper to create a translucent overlay blitting current snapshot beneath
    def _blit_snapshot_with_overlay(self, base_snapshot, overlay_color=(0,0,0), alpha=128):
        # base_snapshot is a Surface already containing whatever background we want
        screen.blit(base_snapshot, (0,0))
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((*overlay_color, alpha))
        screen.blit(overlay, (0,0))

    def run_dev_menu(self):
        # capture snapshot of current screen to show game behind the menu
        try:
            snapshot = screen.copy()
        except Exception:
            # fallback: empty dark background
            snapshot = pygame.Surface((WIDTH, HEIGHT))
            snapshot.fill((28,28,36))
        while True:
            # draw snapshot + 50% overlay so game shows behind
            self._blit_snapshot_with_overlay(snapshot, overlay_color=(0,0,0), alpha=128)
            draw_centered_text(screen, "DEV MENU", pygame.Rect(WIDTH//2 - 220, 40, 440, 60), color=WHITE, font=BIG)
            mx,my = pygame.mouse.get_pos()
            btns = {
                "basehp": pygame.Rect(WIDTH//2 - 140, 140, 280, 40),
                "setspawn": pygame.Rect(WIDTH//2 - 140, 200, 280, 40),
                "setround": pygame.Rect(WIDTH//2 - 140, 260, 280, 40),
                "setmoney": pygame.Rect(WIDTH//2 - 140, 320, 280, 40),
                "settowerlvl": pygame.Rect(WIDTH//2 - 140, 380, 280, 40),
                "debug": pygame.Rect(WIDTH//2 - 140, 440, 280, 40),
                "close": pygame.Rect(WIDTH//2 - 140, 500, 280, 40)
            }
            for k,r in btns.items():
                pygame.draw.rect(screen, LIGHT_GRAY, r)
            draw_centered_text(screen, "Set Base HP", btns["basehp"])
            draw_centered_text(screen, "Spawn One Enemy (type)", btns["setspawn"])
            draw_centered_text(screen, "Set Round", btns["setround"])
            draw_centered_text(screen, "Set Money", btns["setmoney"])
            draw_centered_text(screen, "Set Tower Level (all)", btns["settowerlvl"])
            draw_centered_text(screen, "Debug Dump and Exit", btns["debug"])
            draw_centered_text(screen, "Close", btns["close"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if btns["basehp"].collidepoint(mx,my):
                        val = self.prompt_text("Enter base HP (integer):")
                        if val and val.isdigit(): self.base_health = int(val); self.io_message = "Base HP set"; self.io_message_time = time.time()
                    elif btns["setspawn"].collidepoint(mx,my):
                        # Prompt for type then spawn exactly one enemy of that type immediately
                        val = self.prompt_text("Enter enemy type to spawn now (normal/fast/heavy/boss/shield/stunner):")
                        if val and val.strip():
                            typ = val.strip()
                            # spawn one enemy immediately of requested type (does not affect wave counts)
                            self.spawn_enemy(kind=typ, consume_to_spawn=False)
                            self.io_message = f"Spawned one enemy of type {typ}"
                            self.io_message_time = time.time()
                    elif btns["setround"].collidepoint(mx,my):
                        val = self.prompt_text("Enter round number (integer):")
                        if val and val.isdigit(): self.wave = int(val); self.io_message = "Round set"; self.io_message_time = time.time()
                    elif btns["setmoney"].collidepoint(mx,my):
                        val = self.prompt_text("Enter money amount (integer):")
                        if val and val.isdigit(): self.money = int(val); self.io_message = "Money set"; self.io_message_time = time.time()
                    elif btns["settowerlvl"].collidepoint(mx,my):
                        val = self.prompt_text("Set tower level (integer):")
                        if val and val.isdigit():
                            lvl = int(val)
                            for t in self.towers:
                                t.level = lvl
                                t.set_base_stats()
                                for _ in range(1, lvl):
                                    t.upgrade()
                            self.io_message = "Tower levels set"; self.io_message_time = time.time()
                    elif btns["debug"].collidepoint(mx,my):
                        dump = {
                            "wave": self.wave,
                            "money": self.money,
                            "base_health": self.base_health,
                            "path_cells": self.path_cells,
                            "towers": [{"grid_pos": t.grid_pos, "kind": t.kind, "level": t.level} for t in self.towers],
                            "enemies": [{"index": e.index, "kind": e.kind, "health": e.health} for e in self.enemies]
                        }
                        fname = os.path.join(self.save_dir, f"debug_dump_wave_{self.wave}.json")
                        try:
                            with open(fname, "w", encoding="utf-8") as f:
                                json.dump(dump, f, indent=2)
                        except Exception:
                            pass
                        # set required message then quit
                        self.io_message = "errorcodeno13"
                        self.io_message_time = time.time()
                        pygame.quit(); sys.exit()
                    elif btns["close"].collidepoint(mx,my):
                        return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
            if self.io_message and time.time() - self.io_message_time < 4:
                draw_text(screen, self.io_message, (20, HEIGHT - 40), color=(200,200,50))
            pygame.display.flip(); self.clock.tick(30)

    # ---- prompt helper ----
    def prompt_text(self, prompt):
        text = ""
        active = True
        while active:
            screen.fill((30,30,36))
            draw_text(screen, prompt, (40, 140), color=WHITE, font=BIG)
            box = pygame.Rect(40, 220, WIDTH - 80, 48)
            pygame.draw.rect(screen, WHITE, box)
            pygame.draw.rect(screen, BLACK, box, 2)
            draw_text(screen, text, (box.x + 10, box.y + 10), color=BLACK, font=FONT)
            draw_text(screen, "Enter to confirm, ESC to cancel", (40, 300), color=(200,200,200))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return text.strip()
                    elif event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        if event.unicode and ord(event.unicode) >= 32:
                            text += event.unicode
            pygame.display.flip(); self.clock.tick(30)

    # ---- menus (main/settings/sandbox) ----
    def run_main_menu(self):
        self.level_info = ""
        while self.menu_state == "main":
            screen.fill((28,28,36))
            draw_centered_text(screen, "TOWER DEFENSE", pygame.Rect(WIDTH//2 - 200, 60, 400, 80), color=WHITE, font=BIG)
            mx,my = pygame.mouse.get_pos()
            easy_rect = pygame.Rect(WIDTH//2 - 140, 180, 120, 48)
            hard_rect = pygame.Rect(WIDTH//2 + 20, 180, 120, 48)
            sandbox_rect = pygame.Rect(WIDTH//2 - 60, 250, 120, 48)
            load_rect = pygame.Rect(WIDTH//2 - 110, 320, 220, 40)
            settings_rect = pygame.Rect(WIDTH//2 - 110, 370, 220, 40)
            quit_rect = pygame.Rect(WIDTH//2 - 110, 420, 220, 40)
            pygame.draw.rect(screen, LIGHT_GRAY, easy_rect); pygame.draw.rect(screen, LIGHT_GRAY, hard_rect); pygame.draw.rect(screen, LIGHT_GRAY, sandbox_rect)
            draw_centered_text(screen, "Easy", easy_rect); draw_centered_text(screen, "Hard", hard_rect); draw_centered_text(screen, "Sandbox", sandbox_rect)
            pygame.draw.rect(screen, WHITE, load_rect); pygame.draw.rect(screen, WHITE, settings_rect); pygame.draw.rect(screen, WHITE, quit_rect)
            draw_centered_text(screen, "Load Game", load_rect); draw_centered_text(screen, "Settings", settings_rect); draw_centered_text(screen, "Quit", quit_rect)
            if self.level_info: draw_text(screen, self.level_info, (WIDTH//2 - 160, 520), color=(200,200,200), font=SMALL)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # invisible dev-code area focus in menus too
                    if self.dev_code_rect.collidepoint(event.pos):
                        self.dev_code_focused = True; self.dev_code_buffer = ""; continue
                    else:
                        self.dev_code_focused = False
                    if easy_rect.collidepoint(mx,my):
                        self.reset_state(); self.gen_easy(); self.mode="easy"; self.menu_state="playing"; self.exit_confirm_time=0.0
                        if self.auto_start: self.start_wave(); return
                    elif hard_rect.collidepoint(mx,my):
                        self.reset_state(); self.gen_hard(); self.mode="hard"; self.menu_state="playing"; self.exit_confirm_time=0.0
                        if self.auto_start: self.start_wave(); return
                    elif sandbox_rect.collidepoint(mx,my):
                        self.reset_state(); self.start_sandbox(); self.menu_state="sandbox_draw"; return
                    elif load_rect.collidepoint(mx,my):
                        name = self.prompt_text("Enter .BRIP filename to load (e.g. save1.BRIP):")
                        if name: self.load_game(name)
                    elif settings_rect.collidepoint(mx,my):
                        self.menu_state = "settings"; return
                    elif quit_rect.collidepoint(mx,my):
                        pygame.quit(); sys.exit()
                elif event.type == pygame.KEYDOWN:
                    # hidden dev-code input: accept keystrokes while focused
                    if self.dev_code_focused:
                        if event.key == pygame.K_RETURN:
                            code = self.dev_code_buffer.strip()
                            self.dev_code_focused = False; self.dev_code_buffer = ""
                            if code: self.enter_dev_code_from_buffer(code)
                            continue
                        elif event.key == pygame.K_ESCAPE:
                            self.dev_code_focused = False; self.dev_code_buffer = ""; continue
                        elif event.key == pygame.K_BACKSPACE:
                            self.dev_code_buffer = self.dev_code_buffer[:-1]; continue
                        else:
                            if event.unicode and ord(event.unicode) >= 32:
                                self.dev_code_buffer += event.unicode; continue
                    if event.key == pygame.K_ESCAPE:
                        now = pygame.time.get_ticks()/1000.0
                        if now - self.exit_confirm_time > self.exit_confirm_window:
                            self.exit_confirm_time = now; self.level_info = "Press ESC again to quit"
                        else:
                            pygame.quit(); sys.exit()
                    elif event.key == pygame.K_F1:
                        self.enter_dev_code()
            pygame.display.flip(); self.clock.tick(30)

    def run_settings(self):
        while self.menu_state == "settings":
            screen.fill((24,24,30))
            draw_centered_text(screen, "SETTINGS", pygame.Rect(WIDTH//2 - 180, 60, 360, 60), color=WHITE, font=BIG)
            mx,my = pygame.mouse.get_pos()
            row_x = WIDTH//2 - 160; row_y = 160; row_h=48; row_w=320; gap=16
            auto_rect = pygame.Rect(row_x, row_y, row_w, row_h)
            fs_rect = pygame.Rect(row_x, row_y + (row_h+gap), row_w, row_h)
            sound_rect = pygame.Rect(row_x, row_y + 2*(row_h+gap), row_w, row_h)
            back_rect = pygame.Rect(row_x, row_y + 3*(row_h+gap)+10, row_w, row_h)
            pygame.draw.rect(screen, LIGHT_GRAY if self.auto_start else WHITE, auto_rect); draw_centered_text(screen, f"Auto-start: {'ON' if self.auto_start else 'OFF'}", auto_rect)
            pygame.draw.rect(screen, LIGHT_GRAY if self.fullscreen else WHITE, fs_rect); draw_centered_text(screen, f"Fullscreen: {'ON' if self.fullscreen else 'OFF'}", fs_rect)
            pygame.draw.rect(screen, LIGHT_GRAY if self.sound_on else WHITE, sound_rect); draw_centered_text(screen, f"Sound: {'ON' if self.sound_on else 'OFF'}", sound_rect)
            pygame.draw.rect(screen, WHITE, back_rect); draw_centered_text(screen, "Back", back_rect)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.dev_code_rect.collidepoint(event.pos):
                        self.dev_code_focused = True; self.dev_code_buffer = ""; continue
                    else:
                        self.dev_code_focused = False
                    if auto_rect.collidepoint(mx,my): self.auto_start = not self.auto_start
                    elif fs_rect.collidepoint(mx,my):
                        self.fullscreen = not self.fullscreen
                        if self.fullscreen: pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
                        else: pygame.display.set_mode((WIDTH, HEIGHT))
                    elif sound_rect.collidepoint(mx,my): self.sound_on = not self.sound_on
                    elif back_rect.collidepoint(mx,my): self.menu_state = "main"; return
                elif event.type == pygame.KEYDOWN:
                    if self.dev_code_focused:
                        if event.key == pygame.K_RETURN:
                            code = self.dev_code_buffer.strip()
                            self.dev_code_focused = False; self.dev_code_buffer = ""
                            if code: self.enter_dev_code_from_buffer(code)
                            continue
                        elif event.key == pygame.K_ESCAPE:
                            self.dev_code_focused = False; self.dev_code_buffer = ""; continue
                        elif event.key == pygame.K_BACKSPACE:
                            self.dev_code_buffer = self.dev_code_buffer[:-1]; continue
                        else:
                            if event.unicode and ord(event.unicode) >= 32:
                                self.dev_code_buffer += event.unicode; continue
                    if event.key == pygame.K_ESCAPE: self.menu_state = "main"; return
            pygame.display.flip(); self.clock.tick(30)

    def run_sandbox_draw(self):
        while self.menu_state == "sandbox_draw":
            screen.fill((40,40,48))
            draw_centered_text(screen, "SANDBOX: Draw Path", pygame.Rect(0,12,WIDTH,48), color=WHITE, font=BIG)
            mx,my = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.dev_code_rect.collidepoint(event.pos):
                        self.dev_code_focused = True; self.dev_code_buffer = ""; continue
                    else:
                        self.dev_code_focused = False
                    if event.button == 1:
                        cell = pixel_to_cell(mx,my)
                        if cell and cell[0] < MAP_W: self.sandbox_add_cell(cell)
                    elif event.button == 3: self.sandbox_remove_last()
                elif event.type == pygame.KEYDOWN:
                    if self.dev_code_focused:
                        if event.key == pygame.K_RETURN:
                            code = self.dev_code_buffer.strip()
                            self.dev_code_focused = False; self.dev_code_buffer = ""
                            if code: self.enter_dev_code_from_buffer(code)
                            continue
                        elif event.key == pygame.K_ESCAPE:
                            self.dev_code_focused = False; self.dev_code_buffer = ""; continue
                        elif event.key == pygame.K_BACKSPACE:
                            self.dev_code_buffer = self.dev_code_buffer[:-1]; continue
                        else:
                            if event.unicode and ord(event.unicode) >= 32:
                                self.dev_code_buffer += event.unicode; continue
                    if event.key == pygame.K_RETURN:
                        ok = self.sandbox_finalize()
                        if ok:
                            self.money = 250; self.base_health = 50; self.wave = 1
                            self.towers = []; self.enemies = []; self.projectile_effects = []
                            self.mode = "sandbox"; self.menu_state = "playing"; self.exit_confirm_time = 0.0
                            if self.auto_start: self.start_wave(); return
                    elif event.key == pygame.K_ESCAPE:
                        self.menu_state = "main"; return
                    elif event.key == pygame.K_f:
                        self.fullscreen = not self.fullscreen
                        if self.fullscreen: pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
                        else: pygame.display.set_mode((WIDTH, HEIGHT))
                    elif event.key == pygame.K_s:
                        name = self.prompt_text("Sandbox save filename (no extension):")
                        if name: self.save_sandbox(name + ".bri")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F1:
                        self.enter_dev_code()
            play_surface = screen.subsurface(pygame.Rect(0,0,PLAY_W, PLAY_H))
            play_surface.fill((245,255,250)); self.draw_grid(play_surface)
            for c in self.path_cells:
                rect = pygame.Rect(c[0]*GRID_SIZE, c[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE)
                s = pygame.Surface((GRID_SIZE, GRID_SIZE)); s.set_alpha(180); s.fill((180,180,220)); play_surface.blit(s, rect.topleft)
                px,py = cell_to_pixel(c); pygame.draw.circle(play_surface, (100,100,160), (px,py), 6)
            cell_hover = pixel_to_cell(mx,my)
            if cell_hover and cell_hover[0] < MAP_W:
                rect = pygame.Rect(cell_hover[0]*GRID_SIZE, cell_hover[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE)
                s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA); s.fill((100,200,100,60)); play_surface.blit(s, rect.topleft)
            draw_text(screen, "Left click add; Right click undo; Enter play; S save .bri; F fullscreen", (10, PLAY_H + 6))
            draw_text(screen, self.level_info, (10, PLAY_H + 26))
            self.draw_sidebar(screen); pygame.display.flip(); self.clock.tick(60)

    # ---- Pause menu opened by ESC (preserves state) ----
    def run_pause_menu(self):
        # snapshot current screen so that menu shows over the actual game view
        try:
            snapshot = screen.copy()
        except Exception:
            snapshot = pygame.Surface((WIDTH, HEIGHT)); snapshot.fill((18,18,24))
        while self.menu_state == "pause_menu":
            # draw snapshot + semi-transparent overlay so game shows behind menu
            self._blit_snapshot_with_overlay(snapshot, overlay_color=(0,0,0), alpha=128)
            draw_centered_text(screen, "PAUSED", pygame.Rect(WIDTH//2 - 160, 60, 320, 60), color=WHITE, font=BIG)
            mx,my = pygame.mouse.get_pos()
            resume_rect = pygame.Rect(WIDTH//2 - 110, 180, 220, 48)
            save_rect = pygame.Rect(WIDTH//2 - 110, 250, 220, 44)
            load_rect = pygame.Rect(WIDTH//2 - 110, 305, 220, 44)
            menu_rect = pygame.Rect(WIDTH//2 - 110, 360, 220, 44)
            quit_rect = pygame.Rect(WIDTH//2 - 110, 420, 220, 44)
            pygame.draw.rect(screen, LIGHT_GRAY, resume_rect); pygame.draw.rect(screen, WHITE, save_rect)
            pygame.draw.rect(screen, WHITE, load_rect); pygame.draw.rect(screen, WHITE, menu_rect); pygame.draw.rect(screen, WHITE, quit_rect)
            draw_centered_text(screen, "Resume", resume_rect); draw_centered_text(screen, "Save Game", save_rect)
            draw_centered_text(screen, "Load Game", load_rect); draw_centered_text(screen, "Return to Main Menu", menu_rect)
            draw_centered_text(screen, "Quit", quit_rect)
            draw_text(screen, "ESC or Resume to go back", (WIDTH//2 - 120, HEIGHT - 40))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.dev_code_rect.collidepoint(event.pos):
                        self.dev_code_focused = True; self.dev_code_buffer = ""; continue
                    else:
                        self.dev_code_focused = False
                    if resume_rect.collidepoint(mx,my):
                        self.paused = False; self.menu_state = "playing"; return
                    elif save_rect.collidepoint(mx,my):
                        name = self.prompt_text("Save filename (no extension):")
                        if name: self.save_game(name + ".BRIP")
                    elif load_rect.collidepoint(mx,my):
                        name = self.prompt_text("Load .BRIP filename (e.g. save1.BRIP):")
                        if name: self.load_game(name)
                    elif menu_rect.collidepoint(mx,my):
                        self.menu_state = "main"; self.paused = False; return
                    elif quit_rect.collidepoint(mx,my):
                        pygame.quit(); sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if self.dev_code_focused:
                        if event.key == pygame.K_RETURN:
                            code = self.dev_code_buffer.strip()
                            self.dev_code_focused = False; self.dev_code_buffer = ""
                            if code: self.enter_dev_code_from_buffer(code)
                            continue
                        elif event.key == pygame.K_ESCAPE:
                            self.dev_code_focused = False; self.dev_code_buffer = ""; continue
                        elif event.key == pygame.K_BACKSPACE:
                            self.dev_code_buffer = self.dev_code_buffer[:-1]; continue
                        else:
                            if event.unicode and ord(event.unicode) >= 32:
                                self.dev_code_buffer += event.unicode; continue
                    if event.key == pygame.K_ESCAPE:
                        self.paused = False; self.menu_state = "playing"; return
                    elif event.key == pygame.K_F1:
                        self.enter_dev_code()
            if self.io_message and time.time() - self.io_message_time < 3.5:
                draw_text(screen, self.io_message, (20, HEIGHT - 40), color=(200,200,80))
            pygame.display.flip(); self.clock.tick(30)

    # ---- draw helpers ----
    def draw_grid(self, surf):
        for x in range(MAP_W):
            for y in range(MAP_H):
                rect = pygame.Rect(x*GRID_SIZE, y*GRID_SIZE, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(surf, (230,230,230), rect, 1)

    def draw_path(self, surf):
        if not self.path_pixels: return
        pygame.draw.lines(surf, (180,180,180), False, self.path_pixels, GRID_SIZE//2)
        for p in self.path_pixels: pygame.draw.circle(surf, (150,150,150), p, GRID_SIZE//3)

    def draw_sidebar(self, surf):
        sidebar_rect = pygame.Rect(PLAY_W, 0, WIDTH-PLAY_W, HEIGHT)
        pygame.draw.rect(surf, (245,245,250), sidebar_rect); pygame.draw.rect(surf, BLACK, sidebar_rect, 2)
        x0 = PLAY_W + 10; y = 10
        draw_text(surf, f"Money: ${self.money}", (x0,y)); y+=28
        draw_text(surf, f"Wave: {self.wave}", (x0,y)); y+=28
        draw_text(surf, self.level_info, (x0,y)); y+=40
        draw_text(surf, "Tower Shop:", (x0,y)); y+=8
        for kind, rect, cost, label in self.shop_buttons:
            pygame.draw.rect(surf, LIGHT_GRAY if self.selected_tower_type==kind else WHITE, rect)
            pygame.draw.rect(surf, BLACK, rect, 2)
            draw_text(surf, f"{label} (${cost})", (rect.x+6, rect.y+8))
        y = self.shop_buttons[-1][1].bottom + 10
        draw_text(surf, "Controls:", (x0,y)); y+=20
        draw_text(surf, "Left click to place / draw", (x0,y)); y+=20
        draw_text(surf, "Right click to select / undo", (x0,y)); y+=20
        draw_text(surf, "U upgrade, X sell, P pause", (x0,y)); y+=20
        draw_text(surf, "F fullscreen, ESC pause menu", (x0,y)); y+=20
        draw_text(surf, f"Auto-start: {'ON' if self.auto_start else 'OFF'}", (x0,y)); y+=18
        draw_text(surf, f"Sound: {'ON' if self.sound_on else 'OFF'}", (x0,y))
        # dev button
        if self.show_dev_button:
            pygame.draw.rect(surf, (200,180,80), self.dev_button_rect)
            draw_text(surf, "DEV MENU", (self.dev_button_rect.x + 6, self.dev_button_rect.y + 6))
        # IO message
        if self.io_message and time.time() - self.io_message_time < 3.5:
            draw_text(surf, self.io_message, (x0, HEIGHT - 40), color=(40,40,200))

    def draw_base_box(self, surf):
        box = pygame.Rect(PLAY_W - 110, PLAY_H + 10, 100, 60)
        pygame.draw.rect(surf, BLACK, box)
        draw_text(surf, "BASE", (box.x+10, box.y+6), color=WHITE); draw_text(surf, f"HP: {self.base_health}", (box.x+10, box.y+28), color=WHITE)

    def draw_enemy_hover_info(self, surf, mouse_pos):
        mx,my = mouse_pos
        for e in self.enemies:
            if distance((mx,my),(e.x,e.y)) <= e.radius + 4:
                tx,ty = mx+12, my+12; w,h = 160,60; rect = pygame.Rect(tx,ty,w,h)
                pygame.draw.rect(surf, LIGHT_GRAY, rect); pygame.draw.rect(surf, BLACK, rect,2)
                draw_text(surf, f"Name: {e.name}", (tx+6, ty+6)); draw_text(surf, f"Type: {e.kind}", (tx+6, ty+26))
                draw_text(surf, f"HP: {int(e.health)}/{int(e.max_health)}", (tx+6, ty+42)); break

    # ---- main loop ----
    def run(self):
        while True:
            if self.menu_state == "main":
                self.run_main_menu()
            elif self.menu_state == "settings":
                self.run_settings()
            elif self.menu_state == "sandbox_draw":
                self.run_sandbox_draw()
            elif self.menu_state == "pause_menu":
                self.run_pause_menu()
            elif self.menu_state == "playing":
                if not self.paused and not self.game_over:
                    self.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        # hidden dev-code keyboard capture (only while focused)
                        if self.dev_code_focused:
                            if event.key == pygame.K_RETURN:
                                code = self.dev_code_buffer.strip()
                                self.dev_code_focused = False; self.dev_code_buffer = ""
                                if code: self.enter_dev_code_from_buffer(code)
                                continue
                            elif event.key == pygame.K_ESCAPE:
                                self.dev_code_focused = False; self.dev_code_buffer = ""; continue
                            elif event.key == pygame.K_BACKSPACE:
                                self.dev_code_buffer = self.dev_code_buffer[:-1]; continue
                            else:
                                if event.unicode and ord(event.unicode) >= 32:
                                    self.dev_code_buffer += event.unicode; continue
                        if event.key == pygame.K_SPACE: self.start_wave()
                        elif event.key == pygame.K_u:
                            if self.selected_tower: self.upgrade_tower(self.selected_tower)
                        elif event.key == pygame.K_x:
                            if self.selected_tower: self.sell_tower(self.selected_tower)
                        elif event.key == pygame.K_p:
                            self.paused = True; self.menu_state = "pause_menu"; self.level_info = "Paused"
                        elif event.key == pygame.K_f:
                            self.fullscreen = not self.fullscreen
                            if self.fullscreen: pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
                            else: pygame.display.set_mode((WIDTH, HEIGHT))
                        elif event.key == pygame.K_ESCAPE:
                            # open pause menu rather than clearing state
                            self.paused = True; self.menu_state = "pause_menu"; self.level_info = "Paused (ESC)"
                        elif event.key == pygame.K_F1:
                            self.enter_dev_code()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        mx,my = pygame.mouse.get_pos()
                        # clicking the invisible dev-code area focuses it
                        if self.dev_code_rect.collidepoint(mx,my):
                            self.dev_code_focused = True; self.dev_code_buffer = ""; continue
                        else:
                            # don't forcibly unfocus here; only certain actions clear focus
                            pass
                        # dev button
                        if self.show_dev_button and self.dev_button_rect.collidepoint(mx,my):
                            self.run_dev_menu(); continue
                        if mx >= PLAY_W:
                            for kind, rect, cost, label in self.shop_buttons:
                                if rect.collidepoint(mx,my):
                                    self.selected_tower_type = kind; self.level_info = f"Selected {kind}"; break
                        else:
                            cell = pixel_to_cell(mx,my)
                            if event.button == 1:
                                if self.can_place_at(cell): self.place_tower(cell)
                                else:
                                    clicked_tower = None
                                    for t in self.towers:
                                        rect = pygame.Rect(t.grid_pos[0]*GRID_SIZE, t.grid_pos[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE)
                                        if rect.collidepoint(mx,my): clicked_tower = t; break
                                    self.selected_tower = clicked_tower
                                    if clicked_tower: self.level_info = f"Selected tower (Lv {clicked_tower.level})"
                            elif event.button == 3:
                                clicked_tower = None
                                for t in self.towers:
                                    rect = pygame.Rect(t.grid_pos[0]*GRID_SIZE, t.grid_pos[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE)
                                    if rect.collidepoint(mx,my): clicked_tower = t; break
                                self.selected_tower = clicked_tower
                                if clicked_tower: self.level_info = f"Selected tower (Lv {clicked_tower.level})"
                # draw world
                screen.fill(WHITE)
                play_surface = screen.subsurface(pygame.Rect(0,0,PLAY_W, PLAY_H))
                play_surface.fill((245,255,250)); self.draw_grid(play_surface)
                # draw path
                self.path_pixels = [cell_to_pixel(c) for c in self.path_cells] if self.path_cells else self.path_pixels
                self.draw_path(play_surface)
                for t in self.towers: t.draw(play_surface, selected=(t is self.selected_tower))
                for e in self.enemies: e.draw(play_surface)
                # projectiles
                for p in self.projectile_effects:
                    typ, px, py, life = p
                    if typ == "splash": pygame.draw.circle(play_surface, PURPLE, (int(px), int(py)), 6 + life//2, 2)
                    elif typ == "freeze": pygame.draw.circle(play_surface, LIGHT_GRAY, (int(px), int(py)), 5 + life//3, 2)
                    elif typ == "stun_pulse": pygame.draw.circle(play_surface, (200,180,255), (int(px), int(py)), 6 + life//2, 2)
                    elif typ == "laser": pygame.draw.circle(play_surface, (255,140,120), (int(px), int(py)), 4)
                    elif typ == "mortar": pygame.draw.circle(play_surface, (120,80,40), (int(px), int(py)), 8, 2)
                    else: pygame.draw.circle(play_surface, (240,220,60), (int(px), int(py)), 3)
                self.draw_sidebar(screen); self.draw_base_box(screen)
                mx,my = pygame.mouse.get_pos(); self.draw_enemy_hover_info(screen, (mx,my))
                cell = pixel_to_cell(mx,my)
                if cell:
                    gx,gy = cell
                    rect = pygame.Rect(gx*GRID_SIZE, gy*GRID_SIZE, GRID_SIZE, GRID_SIZE)
                    color = (200,255,200) if self.can_place_at(cell) else (255,200,200)
                    s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA); s.fill((*color, 40)); screen.blit(s, rect.topleft)
                    if self.can_place_at(cell):
                        draw_text(screen, f"${Tower(cell, self.selected_tower_type).cost}", (rect.x+4, rect.y+4))
                if self.selected_tower:
                    t = self.selected_tower
                    panel_rect = pygame.Rect(PLAY_W+10, HEIGHT-140, WIDTH-PLAY_W-20, 130)
                    pygame.draw.rect(screen, (230,230,230), panel_rect); pygame.draw.rect(screen, BLACK, panel_rect,2)
                    draw_text(screen, f"Selected Tower: {t.kind} (Level {t.level})", (PLAY_W+20, HEIGHT-130))
                    draw_text(screen, f"Damage: {t.damage}", (PLAY_W+20, HEIGHT-100))
                    draw_text(screen, f"Range: {t.range}", (PLAY_W+160, HEIGHT-100))
                    draw_text(screen, f"Fire Rate: {t.fire_rate} frames", (PLAY_W+20, HEIGHT-75))
                    draw_text(screen, f"Upgrade Cost: ${t.upgrade_cost}", (PLAY_W+20, HEIGHT-50))
                    draw_text(screen, f"Sell Value: ${int(t.cost * (0.5 + 0.1 * t.level))}", (PLAY_W+180, HEIGHT-50))
                if self.paused:
                    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((0,0,0,120)); screen.blit(overlay, (0,0))
                    draw_text(screen, "PAUSED - ESC or Resume to return", (WIDTH//2-160, HEIGHT//2-10), color=WHITE, font=BIG)
                if self.game_over:
                    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((0,0,0,150)); screen.blit(overlay, (0,0))
                    draw_text(screen, "GAME OVER", (WIDTH//2-80, HEIGHT//2-10), color=WHITE, font=BIG)
                pygame.display.flip(); self.clock.tick(FPS)

if __name__ == "__main__":
    Game().run()