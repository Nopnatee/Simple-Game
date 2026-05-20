import random
import sys
from dataclasses import dataclass

import pygame


WIDTH = 900
HEIGHT = 700
FPS = 60

BG = (8, 10, 18)
GRID = (23, 29, 45)
WHITE = (238, 244, 255)
GREEN = (96, 245, 174)
CYAN = (95, 211, 255)
YELLOW = (255, 220, 92)
RED = (255, 105, 105)
PURPLE = (180, 132, 255)
DIM = (108, 123, 150)

PLAYER_SPEED = 430
PLAYER_FIRE_COOLDOWN = 0.32
PLAYER_BULLET_SPEED = 610
ALIEN_BULLET_SPEED = 265
ALIEN_ROWS = 5
ALIEN_COLS = 11


@dataclass
class Bullet:
    rect: pygame.Rect
    velocity_y: float
    color: tuple[int, int, int]

    def update(self, dt: float) -> None:
        self.rect.y += round(self.velocity_y * dt)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)


@dataclass
class Alien:
    rect: pygame.Rect
    points: int
    color: tuple[int, int, int]

    def draw(self, surface: pygame.Surface, pulse: float) -> None:
        x, y, w, h = self.rect
        bob = int(pulse)
        body = pygame.Rect(x + 4, y + bob + 6, w - 8, h - 10)
        pygame.draw.rect(surface, self.color, body, border_radius=7)
        pygame.draw.rect(surface, WHITE, (x + 10, y + bob + 12, 6, 6), border_radius=2)
        pygame.draw.rect(surface, WHITE, (x + w - 16, y + bob + 12, 6, 6), border_radius=2)
        pygame.draw.rect(surface, BG, (x + 11, y + bob + 14, 3, 3), border_radius=1)
        pygame.draw.rect(surface, BG, (x + w - 15, y + bob + 14, 3, 3), border_radius=1)
        pygame.draw.rect(surface, self.color, (x + 2, y + bob + h - 12, 8, 8), border_radius=3)
        pygame.draw.rect(surface, self.color, (x + w - 10, y + bob + h - 12, 8, 8), border_radius=3)


class Player:
    def __init__(self) -> None:
        self.rect = pygame.Rect(WIDTH // 2 - 26, HEIGHT - 74, 52, 32)
        self.cooldown = 0.0

    def update(self, dt: float, keys: pygame.key.ScancodeWrapper) -> None:
        direction = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            direction -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            direction += 1

        self.rect.x += round(direction * PLAYER_SPEED * dt)
        self.rect.clamp_ip(pygame.Rect(18, 0, WIDTH - 36, HEIGHT))
        self.cooldown = max(0.0, self.cooldown - dt)

    def can_fire(self) -> bool:
        return self.cooldown <= 0

    def fire(self) -> Bullet:
        self.cooldown = PLAYER_FIRE_COOLDOWN
        return Bullet(
            pygame.Rect(self.rect.centerx - 3, self.rect.top - 18, 6, 18),
            -PLAYER_BULLET_SPEED,
            CYAN,
        )

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.polygon(
            surface,
            GREEN,
            [
                (self.rect.centerx, self.rect.top - 12),
                (self.rect.left, self.rect.bottom),
                (self.rect.right, self.rect.bottom),
            ],
        )
        pygame.draw.rect(surface, GREEN, self.rect, border_radius=8)
        pygame.draw.rect(surface, WHITE, (self.rect.centerx - 6, self.rect.top + 8, 12, 9), 2)


class Barrier:
    def __init__(self, x: int, y: int) -> None:
        self.blocks: list[pygame.Rect] = []
        block = 10
        shape = [
            " 111111 ",
            "11111111",
            "11111111",
            "11100111",
            "11000011",
        ]
        for row, line in enumerate(shape):
            for col, marker in enumerate(line):
                if marker == "1":
                    self.blocks.append(pygame.Rect(x + col * block, y + row * block, block, block))

    def hit(self, rect: pygame.Rect) -> bool:
        for block in self.blocks[:]:
            if block.colliderect(rect):
                self.blocks.remove(block)
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        for block in self.blocks:
            pygame.draw.rect(surface, (88, 202, 133), block, border_radius=2)


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Space Invaders")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.big_font = pygame.font.SysFont("consolas", 54, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 16)
        self.reset()

    def reset(self) -> None:
        self.player = Player()
        self.player_bullets: list[Bullet] = []
        self.alien_bullets: list[Bullet] = []
        self.barriers = [Barrier(x, HEIGHT - 188) for x in (115, 310, 505, 700)]
        self.score = 0
        self.lives = 3
        self.level = 1
        self.game_over = False
        self.paused = False
        self.alien_direction = 1
        self.alien_speed = 45
        self.alien_drop = 22
        self.alien_fire_timer = 0.0
        self.alien_fire_delay = 0.95
        self.spawn_wave()

    def spawn_wave(self) -> None:
        self.aliens: list[Alien] = []
        start_x = 102
        start_y = 84
        gap_x = 62
        gap_y = 48
        colors = [PURPLE, RED, YELLOW, CYAN, GREEN]
        points = [40, 30, 20, 20, 10]

        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                rect = pygame.Rect(start_x + col * gap_x, start_y + row * gap_y, 40, 30)
                self.aliens.append(Alien(rect, points[row], colors[row]))

    def run(self) -> None:
        while True:
            dt = self.clock.tick(FPS) / 1000
            self.handle_events()
            if not self.paused and not self.game_over:
                self.update(dt)
            self.draw()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_SPACE and not self.game_over and not self.paused:
                    if self.player.can_fire():
                        self.player_bullets.append(self.player.fire())

    def update(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        self.player.update(dt, keys)

        if keys[pygame.K_SPACE] and self.player.can_fire():
            self.player_bullets.append(self.player.fire())

        self.update_aliens(dt)
        self.update_bullets(dt)
        self.check_collisions()

        if not self.aliens:
            self.level += 1
            self.alien_speed += 16
            self.alien_fire_delay = max(0.32, self.alien_fire_delay * 0.86)
            self.spawn_wave()

    def update_aliens(self, dt: float) -> None:
        if not self.aliens:
            return

        move_x = round(self.alien_direction * self.alien_speed * dt)
        for alien in self.aliens:
            alien.rect.x += move_x

        left = min(alien.rect.left for alien in self.aliens)
        right = max(alien.rect.right for alien in self.aliens)
        if left <= 18 or right >= WIDTH - 18:
            self.alien_direction *= -1
            for alien in self.aliens:
                alien.rect.y += self.alien_drop

        lowest = max(alien.rect.bottom for alien in self.aliens)
        if lowest >= self.player.rect.top:
            self.lose_life()

        self.alien_fire_timer -= dt
        if self.alien_fire_timer <= 0:
            self.fire_alien_bullet()
            self.alien_fire_timer = random.uniform(self.alien_fire_delay * 0.55, self.alien_fire_delay * 1.25)

    def fire_alien_bullet(self) -> None:
        if not self.aliens:
            return

        columns: dict[int, Alien] = {}
        for alien in self.aliens:
            key = alien.rect.centerx // 50
            if key not in columns or alien.rect.bottom > columns[key].rect.bottom:
                columns[key] = alien

        shooter = random.choice(list(columns.values()))
        self.alien_bullets.append(
            Bullet(
                pygame.Rect(shooter.rect.centerx - 3, shooter.rect.bottom + 6, 6, 18),
                ALIEN_BULLET_SPEED + self.level * 14,
                RED,
            )
        )

    def update_bullets(self, dt: float) -> None:
        for bullet in self.player_bullets + self.alien_bullets:
            bullet.update(dt)

        self.player_bullets = [bullet for bullet in self.player_bullets if bullet.rect.bottom > 0]
        self.alien_bullets = [bullet for bullet in self.alien_bullets if bullet.rect.top < HEIGHT]

    def check_collisions(self) -> None:
        for bullet in self.player_bullets[:]:
            if any(barrier.hit(bullet.rect) for barrier in self.barriers):
                self.player_bullets.remove(bullet)
                continue

            for alien in self.aliens[:]:
                if alien.rect.colliderect(bullet.rect):
                    self.score += alien.points
                    self.aliens.remove(alien)
                    if bullet in self.player_bullets:
                        self.player_bullets.remove(bullet)
                    break

        for bullet in self.alien_bullets[:]:
            if any(barrier.hit(bullet.rect) for barrier in self.barriers):
                self.alien_bullets.remove(bullet)
                continue

            if bullet.rect.colliderect(self.player.rect):
                self.alien_bullets.remove(bullet)
                self.lose_life()

    def lose_life(self) -> None:
        self.lives -= 1
        self.alien_bullets.clear()
        self.player_bullets.clear()
        self.player = Player()

        if self.lives <= 0:
            self.game_over = True

    def draw_background(self) -> None:
        self.screen.fill(BG)
        for y in range(40, HEIGHT, 40):
            pygame.draw.line(self.screen, GRID, (0, y), (WIDTH, y))
        for x in range(40, WIDTH, 40):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, HEIGHT))

    def draw_hud(self) -> None:
        hud = f"SCORE {self.score:05d}   LIVES {self.lives}   LEVEL {self.level}"
        text = self.font.render(hud, True, WHITE)
        self.screen.blit(text, (24, 18))
        hint = self.small_font.render("Move: A/D or arrows   Fire: Space   Pause: P   Restart: R", True, DIM)
        self.screen.blit(hint, (24, HEIGHT - 28))

    def draw_overlay(self, title: str, subtitle: str) -> None:
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 148))
        self.screen.blit(shade, (0, 0))

        title_text = self.big_font.render(title, True, WHITE)
        sub_text = self.font.render(subtitle, True, GREEN)
        self.screen.blit(title_text, title_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 36)))
        self.screen.blit(sub_text, sub_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 28)))

    def draw(self) -> None:
        self.draw_background()
        pulse = pygame.time.get_ticks() / 220 % 2

        for barrier in self.barriers:
            barrier.draw(self.screen)
        for alien in self.aliens:
            alien.draw(self.screen, pulse)
        for bullet in self.player_bullets + self.alien_bullets:
            bullet.draw(self.screen)

        self.player.draw(self.screen)
        self.draw_hud()

        if self.paused:
            self.draw_overlay("PAUSED", "Press P to continue")
        if self.game_over:
            self.draw_overlay("GAME OVER", "Press R to restart")

        pygame.display.flip()


if __name__ == "__main__":
    Game().run()
