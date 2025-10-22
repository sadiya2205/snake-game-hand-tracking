import pygame
import cv2
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

class SnakeColorTracking:
    def __init__(self, width=640, height=480, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        
        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game - Color Tracking')
        self.clock = pygame.time.Clock()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Color tracking setup - Track skin color or any colored object
        # HSV range for skin detection (adjust these if needed)
        self.lower_color = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_color = np.array([20, 255, 255], dtype=np.uint8)
        
        # Game variables
        self.reset()
        
        # Tracking variables
        self.hand_position = None
        self.previous_position = None
        self.calibration_mode = True
        self.click_position = None
        self.needs_calibration = False
        
        # Game state
        self.game_started = False
        self.start_button_rect = None
        
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
    
    def calibrate_color(self, event, x, y, flags, param):
        """Mouse callback to calibrate color range"""
        if event == cv2.EVENT_LBUTTONDOWN and self.calibration_mode:
            # Store click position to process in main loop
            self.click_position = (x, y)
            self.needs_calibration = True
    
    def get_direction_from_movement(self):
        """Get direction based on hand movement"""
        if self.hand_position is None or self.previous_position is None:
            return None
        
        dx = self.hand_position[0] - self.previous_position[0]
        dy = self.hand_position[1] - self.previous_position[1]
        
        # Threshold for movement detection
        threshold = 40
        
        if abs(dx) > abs(dy) and abs(dx) > threshold:
            if dx > 0:
                return Direction.RIGHT
            else:
                return Direction.LEFT
        elif abs(dy) > threshold:
            if dy > 0:
                return Direction.DOWN
            else:
                return Direction.UP
        
        return None
    
    def process_tracking(self):
        """Process camera frame for color tracking"""
        success, frame = self.cap.read()
        if not success:
            return None
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Handle calibration click
        if self.needs_calibration and self.click_position:
            x, y = self.click_position
            color = hsv[y, x]
            
            # Set range around selected color
            self.lower_color = np.array([max(0, color[0] - 15), 50, 50], dtype=np.uint8)
            self.upper_color = np.array([min(179, color[0] + 15), 255, 255], dtype=np.uint8)
            
            print(f"Color calibrated! HSV: {color}")
            print(f"Tracking range: {self.lower_color} to {self.upper_color}")
            self.calibration_mode = False
            self.needs_calibration = False
            self.click_position = None
        
        # Create mask for color detection
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate centroid if contour is large enough
            if cv2.contourArea(largest_contour) > 500:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw contour and center
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                    
                    # Update position
                    self.previous_position = self.hand_position
                    self.hand_position = (cx, cy)
        
        # Display instructions on camera frame
        if self.calibration_mode:
            cv2.putText(frame, 'Click on your hand to calibrate', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif not self.game_started:
            cv2.putText(frame, 'Hover hand over START button', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        else:
            cv2.putText(frame, 'Move hand to control snake', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.game_started:
            cv2.putText(frame, f'Score: {self.score}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, 'Q: quit | C: recalibrate', (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show mask in corner
        mask_small = cv2.resize(mask, (160, 120))
        mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[0:120, frame.shape[1]-160:frame.shape[1]] = mask_colored
        
        cv2.imshow('Color Tracking', frame)
        cv2.setMouseCallback('Color Tracking', self.calibrate_color)
        
        return self.get_direction_from_movement()
    
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
        
        if not self.game_started and not self.calibration_mode:
            # Draw start screen
            title_font = pygame.font.Font(None, 72)
            title = title_font.render('SNAKE GAME', True, GREEN)
            title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
            self.display.blit(title, title_rect)
            
            # Draw start button
            button_width = 200
            button_height = 80
            button_x = (self.width - button_width) // 2
            button_y = (self.height - button_height) // 2
            self.start_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            
            # Check if hand is over button
            button_color = GREEN
            if self.check_start_button_click():
                button_color = YELLOW
            
            pygame.draw.rect(self.display, button_color, self.start_button_rect)
            pygame.draw.rect(self.display, WHITE, self.start_button_rect, 3)
            
            button_font = pygame.font.Font(None, 48)
            button_text = button_font.render('START', True, BLACK)
            button_text_rect = button_text.get_rect(center=self.start_button_rect.center)
            self.display.blit(button_text, button_text_rect)
            
            # Instructions
            inst_font = pygame.font.Font(None, 24)
            inst1 = inst_font.render('Hover your hand over START button', True, WHITE)
            inst2 = inst_font.render('to begin the game', True, WHITE)
            self.display.blit(inst1, (self.width // 2 - 150, self.height - 80))
            self.display.blit(inst2, (self.width // 2 - 100, self.height - 50))
            
        else:
            # Draw game elements
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
        
        # Display calibration instruction
        if self.calibration_mode:
            small_font = pygame.font.Font(None, 28)
            instruction = small_font.render('Click on your hand in camera window', True, RED)
            inst_rect = instruction.get_rect(center=(self.width // 2, 30))
            self.display.blit(instruction, inst_rect)
        
        pygame.display.flip()
    
    def play_step(self):
        """Execute one game step"""
        self.frame_iteration += 1
        
        # Process tracking
        new_direction = self.process_tracking()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
                return True
        
        # Check for quit/recalibrate key in OpenCV window
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.cleanup()
            return True
        elif key == ord('c'):
            self.calibration_mode = True
            self.hand_position = None
            self.previous_position = None
            print("Recalibration mode - click on object to track")
        
        # Check for start button hover
        if not self.game_started and not self.calibration_mode:
            if self.check_start_button_click():
                # Add a small counter to prevent instant start
                if not hasattr(self, 'hover_counter'):
                    self.hover_counter = 0
                self.hover_counter += 1
                
                if self.hover_counter > 15:  # Hover for ~1.5 seconds
                    self.game_started = True
                    print("Game Started!")
                    self.hover_counter = 0
            else:
                self.hover_counter = 0
        
        if self.game_started and not self.game_over and not self.calibration_mode:
            # Move snake
            self._move(new_direction)
            self.snake.insert(0, self.head)
            
            # Check collision
            if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
                self.game_over = True
                print(f"Game Over! Final Score: {self.score}")
                return False
            
            # Check if food eaten
            if self.head == self.food:
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()
        
        # Update display
        self._update_ui()
        self.clock.tick(10)  # 10 FPS
        
        return False
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
    
    def run(self):
        """Main game loop"""
        print("Starting Snake Game with Color Tracking!")
        print("1. Click on your hand (or any colored object) in the camera window")
        print("2. Move the tracked object to control the snake")
        print("3. Press 'C' to recalibrate, 'Q' to quit")
        
        running = True
        while running:
            quit_game = self.play_step()
            if quit_game:
                running = False
            
            if self.game_over:
                # Wait for a moment then ask to restart
                pygame.time.wait(2000)
                print("Game Over! Hover over START button to play again or press 'Q' to quit")
                self.game_started = False
                self.game_over = False
                self.reset()
        
        self.cleanup()

if __name__ == '__main__':
    game = SnakeColorTracking()
    game.run()
