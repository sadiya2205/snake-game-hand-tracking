import pygame
import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from collections import namedtuple
import random

# Initialize pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Define directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeHandTracking:
    def __init__(self, width=640, height=480, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        
        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game - Hand Tracking')
        self.clock = pygame.time.Clock()
        
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lower for faster detection
            min_tracking_confidence=0.5,   # Lower for faster tracking
            model_complexity=0  # Fastest model
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera with lower resolution for speed
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Higher FPS
        
        # Game variables
        self.reset()
        
        # Hand tracking variables
        self.hand_center = None
        self.previous_center = None
        
    def reset(self):
        """Reset the game to initial state"""
        self.direction = Direction.RIGHT
        
        self.head = Point(self.width/2, self.height/2)
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - (2 * self.block_size), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.game_over = False
        
    def _place_food(self):
        """Place food at random location"""
        x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def get_hand_direction(self):
        """Get direction matching hand movement instantly"""
        if self.hand_center is None or self.previous_center is None:
            return None
        
        dx = self.hand_center[0] - self.previous_center[0]
        dy = self.hand_center[1] - self.previous_center[1]
        
        # Very low threshold for instant response
        threshold = 8
        
        # Return direction immediately based on largest movement
        if abs(dx) > threshold or abs(dy) > threshold:
            if abs(dx) > abs(dy):
                return Direction.RIGHT if dx > 0 else Direction.LEFT
            else:
                return Direction.DOWN if dy > 0 else Direction.UP
        
        return None
    
    def process_hand_tracking(self):
        """Process camera frame for hand tracking"""
        success, frame = self.cap.read()
        if not success:
            return None
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Resize for faster processing if needed
        # frame = cv2.resize(frame, (320, 240))
        
        # Convert to RGB for MediaPipe (optimize)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Performance boost
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get hand center (palm)
                landmarks = hand_landmarks.landmark
                h, w, c = frame.shape
                
                # Calculate center of palm (using wrist and middle finger base)
                wrist = landmarks[0]
                middle_base = landmarks[9]
                
                center_x = int((wrist.x + middle_base.x) / 2 * w)
                center_y = int((wrist.y + middle_base.y) / 2 * h)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
                
                # Update hand center
                self.previous_center = self.hand_center
                self.hand_center = (center_x, center_y)
        
        # Display instructions on camera frame
        cv2.putText(frame, 'Move hand to control snake', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Score: {self.score}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Hand Tracking', frame)
        
        return self.get_hand_direction()
    
    def is_collision(self, pt=None):
        """Check if collision occurred"""
        if pt is None:
            pt = self.head
        
        # Check boundary collision
        if (pt.x > self.width - self.block_size or pt.x < 0 or 
            pt.y > self.height - self.block_size or pt.y < 0):
            return True
        
        # Check self collision
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _move(self, new_direction):
        """Move the snake based on direction"""
        if new_direction is not None:
            # Prevent moving in opposite direction
            if (new_direction == Direction.RIGHT and self.direction != Direction.LEFT or
                new_direction == Direction.LEFT and self.direction != Direction.RIGHT or
                new_direction == Direction.UP and self.direction != Direction.DOWN or
                new_direction == Direction.DOWN and self.direction != Direction.UP):
                self.direction = new_direction
        
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
    
    def _update_ui(self):
        """Update the game display"""
        self.display.fill(BLACK)
        
        # Draw snake
        for i, pt in enumerate(self.snake):
            color = BLUE if i == 0 else GREEN
            pygame.draw.rect(self.display, color, 
                           pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            if i == 0:
                pygame.draw.rect(self.display, WHITE, 
                               pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Draw food
        pygame.draw.rect(self.display, RED, 
                        pygame.Rect(self.food.x, self.food.y, 
                                  self.block_size, self.block_size))
        
        # Display score
        font = pygame.font.Font(None, 36)
        text = font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [10, 10])
        
        # Display instructions and direction indicator
        small_font = pygame.font.Font(None, 24)
        instruction = small_font.render('Move your hand to control the snake', True, YELLOW)
        self.display.blit(instruction, [10, self.height - 30])
        
        # Show current direction
        dir_font = pygame.font.Font(None, 28)
        dir_text = f'Direction: {self.direction.name}'
        dir_render = dir_font.render(dir_text, True, GREEN)
        self.display.blit(dir_render, [self.width - 200, 10])
        
        pygame.display.flip()
    
    def play_step(self):
        """Execute one game step"""
        self.frame_iteration += 1
        
        # Process hand tracking
        new_direction = self.process_hand_tracking()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
                return True
        
        # Check for quit key in OpenCV window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()
            return True
        
        if not self.game_over:
            # Move snake
            self._move(new_direction)
            self.snake.insert(0, self.head)
            
            # Check collision
            if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
                self.game_over = True
                return False
            
            # Check if food eaten
            if self.head == self.food:
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()
            
            # Update display
            self._update_ui()
            self.clock.tick(15)  # 15 FPS for faster response
        
        return False
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
    
    def run(self):
        """Main game loop"""
        print("Starting Snake Game with Hand Tracking!")
        print("Move your hand left/right/up/down to control the snake")
        print("Press 'Q' to quit or 'R' to restart")
        
        running = True
        while running:
            quit_game = self.play_step()
            if quit_game:
                running = False
            
            if self.game_over:
                # Display game over and wait for restart
                font = pygame.font.Font(None, 72)
                game_over_text = font.render('GAME OVER', True, RED)
                text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2))
                self.display.blit(game_over_text, text_rect)
                
                small_font = pygame.font.Font(None, 36)
                restart_text = small_font.render('Press R to Restart', True, WHITE)
                restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 50))
                self.display.blit(restart_text, restart_rect)
                pygame.display.flip()
                
                print(f"Game Over! Final Score: {self.score}")
                print("Press 'R' to restart or 'Q' to quit")
                
                waiting = True
                while waiting:
                    # Keep camera active
                    self.cap.read()
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r:
                                self.reset()
                                self.game_over = False
                                waiting = False
                            elif event.key == pygame.K_q:
                                running = False
                                waiting = False
                    
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('r'):
                        self.reset()
                        self.game_over = False
                        waiting = False
                    elif key == ord('q'):
                        running = False
                        waiting = False
        
        self.cleanup()

if __name__ == '__main__':
    game = SnakeHandTracking()
    game.run()
