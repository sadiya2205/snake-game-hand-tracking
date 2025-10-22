import pygame
import cv2
import numpy as np
from enum import Enum
from collections import namedtuple
import random
import math

# Initialize pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Define directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGestureControl:
    def __init__(self, width=640, height=480, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        
        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game - Gesture Control')
        self.clock = pygame.time.Clock()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Color tracking for two fingers
        # Thumb color (e.g., red sticker)
        self.thumb_lower = np.array([0, 100, 100], dtype=np.uint8)
        self.thumb_upper = np.array([10, 255, 255], dtype=np.uint8)
        
        # Pointer finger color (e.g., blue sticker)
        self.pointer_lower = np.array([100, 100, 100], dtype=np.uint8)
        self.pointer_upper = np.array([130, 255, 255], dtype=np.uint8)
        
        # Game variables
        self.reset()
        
        # Tracking variables
        self.thumb_position = None
        self.pointer_position = None
        self.previous_thumb = None
        self.previous_pointer = None
        self.fingers_pinched = False
        self.last_direction = None
        
        # Calibration
        self.calibration_mode = True
        self.calibration_step = 0  # 0: thumb, 1: pointer
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
            self.click_position = (x, y)
            self.needs_calibration = True
    
    def check_fingers_pinched(self):
        """Check if thumb and pointer finger are close (pinched)"""
        if self.thumb_position and self.pointer_position:
            distance = math.sqrt(
                (self.thumb_position[0] - self.pointer_position[0])**2 + 
                (self.thumb_position[1] - self.pointer_position[1])**2
            )
            # If distance is less than threshold, fingers are pinched
            return distance < 50
        return False
    
    def get_pinch_direction(self):
        """Get direction when fingers are pinched and moving"""
        if not self.fingers_pinched:
            return None
        
        if self.thumb_position and self.previous_thumb:
            # Use thumb movement for direction
            dx = self.thumb_position[0] - self.previous_thumb[0]
            dy = self.thumb_position[1] - self.previous_thumb[1]
            
            # Threshold for movement detection
            threshold = 25
            
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
        """Process camera frame for two-finger tracking"""
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
            
            if self.calibration_step == 0:
                # Calibrate thumb
                self.thumb_lower = np.array([max(0, color[0] - 15), 50, 50], dtype=np.uint8)
                self.thumb_upper = np.array([min(179, color[0] + 15), 255, 255], dtype=np.uint8)
                print(f"Thumb color calibrated! HSV: {color}")
                self.calibration_step = 1
            else:
                # Calibrate pointer
                self.pointer_lower = np.array([max(0, color[0] - 15), 50, 50], dtype=np.uint8)
                self.pointer_upper = np.array([min(179, color[0] + 15), 255, 255], dtype=np.uint8)
                print(f"Pointer color calibrated! HSV: {color}")
                self.calibration_mode = False
                self.calibration_step = 0
            
            self.needs_calibration = False
            self.click_position = None
        
        # Track thumb (red/color 1)
        thumb_mask = cv2.inRange(hsv, self.thumb_lower, self.thumb_upper)
        thumb_mask = cv2.erode(thumb_mask, np.ones((5, 5), np.uint8), iterations=2)
        thumb_mask = cv2.dilate(thumb_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        # Track pointer (blue/color 2)
        pointer_mask = cv2.inRange(hsv, self.pointer_lower, self.pointer_upper)
        pointer_mask = cv2.erode(pointer_mask, np.ones((5, 5), np.uint8), iterations=2)
        pointer_mask = cv2.dilate(pointer_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        # Find thumb contour
        thumb_contours, _ = cv2.findContours(thumb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if thumb_contours:
            largest = max(thumb_contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 300:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
                    cv2.putText(frame, 'THUMB', (cx-30, cy-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    self.previous_thumb = self.thumb_position
                    self.thumb_position = (cx, cy)
        
        # Find pointer contour
        pointer_contours, _ = cv2.findContours(pointer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if pointer_contours:
            largest = max(pointer_contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 300:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 0), -1)
                    cv2.putText(frame, 'POINTER', (cx-35, cy-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    self.previous_pointer = self.pointer_position
                    self.pointer_position = (cx, cy)
        
        # Check if fingers are pinched
        self.fingers_pinched = self.check_fingers_pinched()
        
        # Draw line between fingers if both detected
        if self.thumb_position and self.pointer_position:
            color = (0, 255, 0) if self.fingers_pinched else (255, 255, 255)
            cv2.line(frame, self.thumb_position, self.pointer_position, color, 2)
            
            # Display distance
            distance = math.sqrt(
                (self.thumb_position[0] - self.pointer_position[0])**2 + 
                (self.thumb_position[1] - self.pointer_position[1])**2
            )
            mid_x = (self.thumb_position[0] + self.pointer_position[0]) // 2
            mid_y = (self.thumb_position[1] + self.pointer_position[1]) // 2
            cv2.putText(frame, f'{int(distance)}', (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display instructions
        if self.calibration_mode:
            if self.calibration_step == 0:
                cv2.putText(frame, 'Click on THUMB (red sticker/marker)', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'Click on POINTER finger (blue sticker/marker)', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif not self.game_started:
            cv2.putText(frame, 'Pinch fingers & hover over START', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        else:
            status = 'PINCHED - Move to control!' if self.fingers_pinched else 'Pinch fingers together'
            color = (0, 255, 0) if self.fingers_pinched else (0, 165, 255)
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.game_started:
            cv2.putText(frame, f'Score: {self.score}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, 'Q: quit | C: recalibrate', (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show masks in corners
        thumb_small = cv2.resize(thumb_mask, (120, 90))
        pointer_small = cv2.resize(pointer_mask, (120, 90))
        thumb_colored = cv2.cvtColor(thumb_small, cv2.COLOR_GRAY2BGR)
        pointer_colored = cv2.cvtColor(pointer_small, cv2.COLOR_GRAY2BGR)
        frame[0:90, frame.shape[1]-120:frame.shape[1]] = thumb_colored
        frame[100:190, frame.shape[1]-120:frame.shape[1]] = pointer_colored
        
        cv2.imshow('Gesture Control', frame)
        cv2.setMouseCallback('Gesture Control', self.calibrate_color)
        
        return self.get_pinch_direction()
    
    def check_start_button_click(self):
        """Check if pinched fingers are over start button"""
        if self.fingers_pinched and self.thumb_position and self.start_button_rect:
            game_x = self.thumb_position[0]
            game_y = self.thumb_position[1]
            
            if self.start_button_rect.collidepoint(game_x, game_y):
                return True
        return False
    
    def is_collision(self, pt=None):
        """Check if collision occurred"""
        if pt is None:
            pt = self.head
        
        if (pt.x > self.width - self.block_size or pt.x < 0 or 
            pt.y > self.height - self.block_size or pt.y < 0):
            return True
        
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
            inst1 = inst_font.render('Pinch thumb & pointer finger together', True, WHITE)
            inst2 = inst_font.render('Hover over START button to begin', True, WHITE)
            self.display.blit(inst1, (self.width // 2 - 165, self.height - 80))
            self.display.blit(inst2, (self.width // 2 - 150, self.height - 50))
            
        else:
            # Draw game elements
            for i, pt in enumerate(self.snake):
                color = BLUE if i == 0 else GREEN
                pygame.draw.rect(self.display, color, 
                               pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
                if i == 0:
                    pygame.draw.rect(self.display, WHITE, 
                                   pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
            pygame.draw.rect(self.display, RED, 
                            pygame.Rect(self.food.x, self.food.y, 
                                      self.block_size, self.block_size))
            
            font = pygame.font.Font(None, 36)
            text = font.render(f'Score: {self.score}', True, WHITE)
            self.display.blit(text, [10, 10])
            
            # Pinch status
            small_font = pygame.font.Font(None, 24)
            status_text = 'PINCHED!' if self.fingers_pinched else 'Pinch to move'
            status_color = GREEN if self.fingers_pinched else ORANGE
            status = small_font.render(status_text, True, status_color)
            self.display.blit(status, [self.width - 150, 10])
        
        if self.calibration_mode:
            font = pygame.font.Font(None, 28)
            if self.calibration_step == 0:
                text = font.render('Click on THUMB in camera window', True, RED)
            else:
                text = font.render('Click on POINTER FINGER in camera window', True, BLUE)
            text_rect = text.get_rect(center=(self.width // 2, 30))
            self.display.blit(text, text_rect)
        
        pygame.display.flip()
    
    def play_step(self):
        """Execute one game step"""
        self.frame_iteration += 1
        
        new_direction = self.process_tracking()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
                return True
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.cleanup()
            return True
        elif key == ord('c'):
            self.calibration_mode = True
            self.calibration_step = 0
            self.thumb_position = None
            self.pointer_position = None
            print("Recalibration mode - click on thumb first, then pointer finger")
        
        # Check for start button
        if not self.game_started and not self.calibration_mode:
            if self.check_start_button_click():
                if not hasattr(self, 'hover_counter'):
                    self.hover_counter = 0
                self.hover_counter += 1
                
                if self.hover_counter > 15:
                    self.game_started = True
                    print("Game Started!")
                    self.hover_counter = 0
            else:
                self.hover_counter = 0
        
        if self.game_started and not self.game_over and not self.calibration_mode:
            self._move(new_direction)
            self.snake.insert(0, self.head)
            
            if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
                self.game_over = True
                print(f"Game Over! Final Score: {self.score}")
                return False
            
            if self.head == self.food:
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()
        
        self._update_ui()
        self.clock.tick(10)
        
        return False
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
    
    def run(self):
        """Main game loop"""
        print("Starting Snake Game with Gesture Control!")
        print("=" * 50)
        print("SETUP:")
        print("1. Put colored stickers on your thumb and pointer finger")
        print("   - Thumb: Red sticker (or any bright color)")
        print("   - Pointer: Blue sticker (or different color)")
        print("2. Click on thumb in camera window")
        print("3. Click on pointer finger in camera window")
        print("\nPLAY:")
        print("- Pinch thumb & pointer together")
        print("- While pinched, move your hand to control snake")
        print("- Hover pinched fingers over START to begin")
        print("=" * 50)
        
        running = True
        while running:
            quit_game = self.play_step()
            if quit_game:
                running = False
            
            if self.game_over:
                pygame.time.wait(2000)
                print("Game Over! Pinch and hover over START to play again")
                self.game_started = False
                self.game_over = False
                self.reset()
        
        self.cleanup()

if __name__ == '__main__':
    game = SnakeGestureControl()
    game.run()
