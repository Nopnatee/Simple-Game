import random
import sys

import pygame


WIDTH = 960
HEIGHT = 540
FPS = 60

PADDLE_WIDTH = 16
PADDLE_HEIGHT = 104
PADDLE_MARGIN = 42
PADDLE_SPEED = 400

BALL_SIZE = 16
BALL_SPEED_START = 430
BALL_SPEED_LIMIT = 760
BALL_SPEEDUP = 1.05

WINNING_SCORE = 10

BACKGROUND = (13, 16, 24)
FOREGROUND = (236, 239, 244)
MUTED = (94, 103, 120)
ACCENT = (112, 194, 255)


class Paddle:
    def __init__(self, x):
        self.rect = pygame.Rect(x, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.score = 0

    def move(self, direction, dt):
        self.rect.y += int(direction * PADDLE_SPEED * dt)
        self.rect.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))


class Ball:
    def __init__(self):
        self.rect = pygame.Rect(0, 0, BALL_SIZE, BALL_SIZE)
        self.velocity = pygame.Vector2()
        self.reset(random.choice((-1, 1)))

    def reset(self, direction):
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        angle = random.uniform(-0.55, 0.55)
        self.velocity.update(direction * BALL_SPEED_START, angle * BALL_SPEED_START)

    def update(self, dt):
        self.rect.x += int(self.velocity.x * dt)
        self.rect.y += int(self.velocity.y * dt)

        if self.rect.top <= 0:
            self.rect.top = 0
            self.velocity.y *= -1
        elif self.rect.bottom >= HEIGHT:
            self.rect.bottom = HEIGHT
            self.velocity.y *= -1

    def bounce_from(self, paddle):
        if not self.rect.colliderect(paddle.rect):
            return

        if self.velocity.x < 0:
            self.rect.left = paddle.rect.right
        else:
            self.rect.right = paddle.rect.left

        offset = (self.rect.centery - paddle.rect.centery) / (PADDLE_HEIGHT / 2)
        speed = min(self.velocity.length() * BALL_SPEEDUP, BALL_SPEED_LIMIT)
        self.velocity.x = speed if self.velocity.x < 0 else -speed
        self.velocity.y = offset * speed * 0.85


def draw_center_line(surface):
    dash_height = 18
    gap = 14
    x = WIDTH // 2 - 2

    for y in range(0, HEIGHT, dash_height + gap):
        pygame.draw.rect(surface, MUTED, (x, y, 4, dash_height), border_radius=2)


def draw_text(surface, font, text, center, color=FOREGROUND):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=center)
    surface.blit(rendered, rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Two Player Pong")
    clock = pygame.time.Clock()

    score_font = pygame.font.SysFont("consolas", 72, bold=True)
    small_font = pygame.font.SysFont("consolas", 20)

    left = Paddle(PADDLE_MARGIN)
    right = Paddle(WIDTH - PADDLE_MARGIN - PADDLE_WIDTH)
    ball = Ball()
    paused = False

    while True:
        dt = clock.tick(FPS) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    left.score = 0
                    right.score = 0
                    ball.reset(random.choice((-1, 1)))
                    paused = False

        keys = pygame.key.get_pressed()

        left_direction = 0
        if keys[pygame.K_w]:
            left_direction -= 1
        if keys[pygame.K_s]:
            left_direction += 1

        right_direction = 0
        if keys[pygame.K_UP]:
            right_direction -= 1
        if keys[pygame.K_DOWN]:
            right_direction += 1

        if not paused and left.score < WINNING_SCORE and right.score < WINNING_SCORE:
            left.move(left_direction, dt)
            right.move(right_direction, dt)
            ball.update(dt)
            ball.bounce_from(left)
            ball.bounce_from(right)

            if ball.rect.right < 0:
                right.score += 1
                ball.reset(-1)
            elif ball.rect.left > WIDTH:
                left.score += 1
                ball.reset(1)

        screen.fill(BACKGROUND)
        draw_center_line(screen)

        pygame.draw.rect(screen, FOREGROUND, left.rect, border_radius=6)
        pygame.draw.rect(screen, FOREGROUND, right.rect, border_radius=6)
        pygame.draw.ellipse(screen, ACCENT, ball.rect)

        draw_text(screen, score_font, str(left.score), (WIDTH // 2 - 120, 72))
        draw_text(screen, score_font, str(right.score), (WIDTH // 2 + 120, 72))

        if left.score >= WINNING_SCORE:
            draw_text(screen, small_font, "LEFT PLAYER WINS - PRESS R", (WIDTH // 2, HEIGHT - 40), ACCENT)
        elif right.score >= WINNING_SCORE:
            draw_text(screen, small_font, "RIGHT PLAYER WINS - PRESS R", (WIDTH // 2, HEIGHT - 40), ACCENT)
        elif paused:
            draw_text(screen, small_font, "PAUSED - SPACE TO RESUME", (WIDTH // 2, HEIGHT - 40), ACCENT)
        else:
            draw_text(screen, small_font, "W/S     SPACE PAUSE     UP/DOWN", (WIDTH // 2, HEIGHT - 40), MUTED)

        pygame.display.flip()


if __name__ == "__main__":
    main()
