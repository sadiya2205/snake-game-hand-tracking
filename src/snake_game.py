import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

# Initialize pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Define directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Snake Game Class
class SnakeGame:
    def __init__(self, width=640, height=480, block_size=20, speed=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed
        
        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game AI')
        self.clock = pygame.time.Clock()
        
        # Initialize game state
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        self.direction = Direction.RIGHT
        
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x - self.block_size, self.head.y),
                      Point(self.head.x - (2 * self.block_size), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        return self.get_state()
    
    def _place_food(self):
        """Place food at random location"""
        x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def play_step(self, action):
        """Execute one game step"""
        self.frame_iteration += 1
        
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move
        self._move(action)  # Update the head
        self.snake.insert(0, self.head)
        
        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)
        
        # 6. Return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        """Check if collision occurred"""
        if pt is None:
            pt = self.head
        
        # Check boundary collision
        if pt.x > self.width - self.block_size or pt.x < 0 or pt.y > self.height - self.block_size or pt.y < 0:
            return True
        
        # Check self collision
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _update_ui(self):
        """Update the game display"""
        self.display.fill(BLACK)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        
        # Display score
        font = pygame.font.Font(None, 36)
        text = font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()
    
    def _move(self, action):
        """Move the snake based on action"""
        # action = [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
        
        self.head = Point(x, y)
    
    def get_state(self):
        """Get current game state as numpy array"""
        head = self.snake[0]
        point_l = Point(head.x - self.block_size, head.y)
        point_r = Point(head.x + self.block_size, head.y)
        point_u = Point(head.x, head.y - self.block_size)
        point_d = Point(head.x, head.y + self.block_size)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),
            
            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),
            
            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y   # food down
        ]
        
        return np.array(state, dtype=int)