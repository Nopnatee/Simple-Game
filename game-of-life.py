# life_gpu.py
import sys
import argparse

import pygame
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# Config (tweakable via CLI)
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GPU Game of Life (PyTorch + Pygame)")
    p.add_argument("--width", type=int, default=160, help="Grid width (cells)")
    p.add_argument("--height", type=int, default=100, help="Grid height (cells)")
    p.add_argument("--cell", type=int, default=6, help="Cell size in pixels")
    p.add_argument("--fps", type=int, default=60, help="Target FPS")
    p.add_argument("--updates-per-frame", type=int, default=1, help="Simulation steps per frame")
    p.add_argument("--density", type=float, default=0.2, help="Initial live cell probability")
    p.add_argument("--wrap", action="store_true", help="Toroidal edges (wrap around)")
    return p.parse_args()

# ------------------------------------------------------------
# Game of Life engine (GPU)
# ------------------------------------------------------------
class LifeGPU:
    def __init__(self, W, H, wrap=True, device=None):
        self.W = W
        self.H = H
        self.wrap = wrap

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # State tensor shape: (1, 1, H, W) for conv2d convenience
        self.state = torch.zeros((1, 1, H, W), dtype=torch.uint8, device=self.device)

        # 3x3 kernel of ones except center 0 to count neighbors
        k = torch.ones((1, 1, 3, 3), dtype=torch.uint8)
        k[0, 0, 1, 1] = 0
        self.kernel = k.to(self.device)

    @torch.no_grad()
    def randomize(self, p=0.2):
        rand = torch.rand((1, 1, self.H, self.W), device=self.device)
        self.state.copy_((rand < p).to(torch.uint8))

    @torch.no_grad()
    def clear(self):
        self.state.zero_()

    @torch.no_grad()
    def toggle_cell(self, x, y):
        # x in [0,W), y in [0,H)
        self.state[0, 0, y % self.H, x % self.W] ^= 1

    @torch.no_grad()
    def step(self, steps=1):
        for _ in range(steps):
            if self.wrap:
                padded = F.pad(self.state, (1, 1, 1, 1), mode="circular")
            else:
                padded = F.pad(self.state, (1, 1, 1, 1), mode="constant", value=0)

            # neighbor count via conv (float math), then cast to int for equality checks
            neigh = F.conv2d(padded.float(), self.kernel.float()).to(torch.int8)

            alive = self.state.to(torch.uint8)
            survive = ((alive == 1) & ((neigh == 2) | (neigh == 3))).to(torch.uint8)
            born    = ((alive == 0) & (neigh == 3)).to(torch.uint8)

            self.state = (survive | born).to(torch.uint8).reshape(1, 1, self.H, self.W)

# ------------------------------------------------------------
# Pygame renderer
# ------------------------------------------------------------
class Renderer:
    def __init__(self, life: LifeGPU, cell_px=6):
        self.life = life
        self.cell_px = max(2, int(cell_px))
        self.W = life.W
        self.H = life.H
        self.surface = pygame.display.set_mode((self.W * self.cell_px, self.H * self.cell_px))
        pygame.display.set_caption(f"GPU Game of Life [{life.device.type.upper()}]")

        self.dead_color = (10, 10, 16)
        self.alive_color = (240, 240, 240)

        self.clock = pygame.time.Clock()

    def resize_cell(self, cell_px):
        self.cell_px = max(2, int(cell_px))
        self.surface = pygame.display.set_mode((self.W * self.cell_px, self.H * self.cell_px))

    def draw(self):
        # state: (1,1,H,W) -> (H,W) uint8 on CPU
        grid = self.life.state.squeeze().to(dtype=torch.uint8, device="cpu").numpy()  # (H, W)

        # Pygame's surfarray uses (W, H, 3), so transpose
        grid_T = grid.T  # (W, H)

        # Build RGB image with vectorized masking
        img = np.empty((self.W, self.H, 3), dtype=np.uint8)
        img[:] = self.dead_color
        img[grid_T == 1] = self.alive_color

        # Make a tiny surface and scale it up
        small = pygame.surfarray.make_surface(img)  # (W,H,3)
        scaled = pygame.transform.scale(small, (self.W * self.cell_px, self.H * self.cell_px))
        self.surface.blit(scaled, (0, 0))
        pygame.display.flip()

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main():
    args = parse_args()
    pygame.init()

    life = LifeGPU(args.width, args.height, wrap=args.wrap)
    if life.device.type != "cuda":
        print("[Heads-up] CUDA not detected. Running on CPU. Install the CUDA build of PyTorch for GPU acceleration.")

    life.randomize(args.density)
    renderer = Renderer(life, cell_px=args.cell)

    paused = False
    updates_per_frame = max(1, args.updates_per_frame)
    running = True

    font = pygame.font.SysFont("consolas", 16)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    life.randomize(args.density)
                elif event.key == pygame.K_c:
                    life.clear()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    updates_per_frame = min(128, updates_per_frame * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    updates_per_frame = max(1, updates_per_frame // 2)
                elif event.key == pygame.K_1:
                    renderer.resize_cell(max(2, renderer.cell_px - 1))
                elif event.key == pygame.K_2:
                    renderer.resize_cell(min(32, renderer.cell_px + 1))

            elif event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and event.buttons[0]):
                mx, my = pygame.mouse.get_pos()
                x = mx // renderer.cell_px
                y = my // renderer.cell_px
                if 0 <= x < life.W and 0 <= y < life.H:
                    life.toggle_cell(x, y)

        if not paused:
            life.step(steps=updates_per_frame)

        renderer.draw()

        # HUD
        hud = f"{'PAUSED' if paused else 'RUNNING'}  |  {life.device.type.upper()}  |  UPS x{updates_per_frame}"
        text = font.render(hud, True, (160, 220, 160))
        renderer.surface.blit(text, (8, 8))
        pygame.display.update((0, 0, 320, 32))

        renderer.clock.tick(args.fps/4)

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
