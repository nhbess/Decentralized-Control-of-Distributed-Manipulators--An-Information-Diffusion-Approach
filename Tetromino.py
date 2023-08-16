import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

class Tetromino:
    def __init__(self, category: str, TILE_SIZE: int, resolution: int):
        self.TILE_SIZE = TILE_SIZE
        if category not in ["I", "O", "T", "J", "L", "S", "Z"]:
            raise ValueError(
                "Invalid category, must be one of: I, O, T, J, L, S, Z")

        sizes = {"I": (resolution*self.TILE_SIZE, resolution*self.TILE_SIZE*4),
                 "L": (resolution*self.TILE_SIZE*2, resolution*self.TILE_SIZE*3), 
                 "J": (resolution*self.TILE_SIZE*2, resolution*self.TILE_SIZE*3),
                 "O": (resolution*self.TILE_SIZE*2, resolution*self.TILE_SIZE*2), 
                 "S": (resolution*self.TILE_SIZE*2, resolution*self.TILE_SIZE*3), 
                 "Z": (resolution*self.TILE_SIZE*2, resolution*self.TILE_SIZE*3), 
                 "T": (resolution*self.TILE_SIZE*2, resolution*self.TILE_SIZE*3)}

        self.surface = pygame.transform.scale(pygame.image.load(f"Images/Tetrominos/{category}.png"), size=sizes[category])
        self.rect = self.surface.get_rect()
        self.mask = pygame.mask.from_surface(self.surface)
        self.angle = 0
        self.rect.x = -self.rect.width//2
        self.rect.y = -self.rect.height//2
        self.max_speed = 2
        self.max_rotation = 2
        self.surface_to_draw = self.surface

        self.center = self.get_geometric_center()
        self.color = (255, 159, 240)

    def set_angle(self, angle: int):
        self.angle = angle
        self.surface = pygame.transform.rotate(self.surface, self.angle)
        self.rect = self.surface.get_rect()
        self.mask = pygame.mask.from_surface(self.surface)

    def get_geometric_center(self) -> np.array:
        return np.array(self.mask.centroid()) + np.array(self.rect.topleft)
    
    def get_angle(self) -> int:
        return self.angle

    def move(self, vector: np.array):
        length = np.linalg.norm(vector)
        if length == 0:
            return
        vector = vector / length
        vector = vector * self.max_speed
        self.rect.center += vector
        self.center = self.get_geometric_center()

    def rotate(self, angle: int = 0, allow_max_rotation: bool = True):
        if allow_max_rotation:
            angle = max(min(angle, self.max_rotation), -self.max_rotation)
        self.angle = (self.angle + angle)
        position = self.rect.center

        self.surface_to_draw = pygame.transform.rotate(self.surface, self.angle)
        self.rect = self.surface_to_draw.get_rect()
        self.mask = pygame.mask.from_surface(self.surface_to_draw)
        self.rect.center = position
        self.center = self.get_geometric_center()

    def draw(self, window: pygame.Surface):
        window.blit(self.surface_to_draw, (self.rect.center[0] - int(self.surface_to_draw.get_width() / 2), self.rect.center[1] - int(self.surface_to_draw.get_height() / 2)))
        #pygame.draw.rect(window, (255, 0, 0), self.rect, 1)
    
    def draw_contour(self, window: pygame.Surface):
        for point in self.mask.outline():
            pygame.draw.circle(window, self.color, (self.rect.x + point[0], self.rect.y + point[1]), 1)
        pygame.draw.circle(window, self.color, self.center, self.TILE_SIZE//10)
        pygame.draw.line(window, self.color, self.center, (self.center[0] + self.TILE_SIZE * np.cos(np.radians(-self.angle)), self.center[1] + self.TILE_SIZE * np.sin(np.radians(-self.angle))), 1)
    
if __name__ == "__main__":
    pass
