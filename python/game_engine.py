import pygame
import numpy as np
import sys
import os
import glob
import random

from game_env import StoneGameEnv

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (240, 240, 240)
STONE_COLOR = (100, 100, 100)
TEXT_COLOR = (20, 20, 20)
BUTTON_COLOR = (200, 200, 200)
BUTTON_HOVER_COLOR = (180, 180, 180)
FONT_SIZE = 32

class GameEngine:
    def __init__(self, q_table_path=None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("21 Stones Game")
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.clock = pygame.time.Clock()

        self.env = StoneGameEnv()
        
        self.q_table = None
        if q_table_path:
            if os.path.exists(q_table_path):
                self.q_table = np.load(q_table_path)
                print(f"Q-table loaded successfully from {q_table_path}")
            else:
                print(f"Warning: Q-table path not found at {q_table_path}. AI will use basic logic.")

        # Game state management
        self.running = True
        self.game_state = "menu"  # "menu", "playing", "game_over"
        self.winner = None

        # Button setup
        self.buttons = {}
        button_width, button_height = 150, 50
        button_y = SCREEN_HEIGHT - 100
        button_actions = ["Take 1", "Take 2", "Take 3"]
        total_button_width = len(button_actions) * button_width + (len(button_actions) - 1) * 20
        start_x = (SCREEN_WIDTH - total_button_width) // 2
        for i, text in enumerate(button_actions):
            rect = pygame.Rect(start_x + i * (button_width + 20), button_y, button_width, button_height)
            self.buttons[i + 1] = {"rect": rect, "text": text}

    def run(self):
        while self.running:
            if self.game_state == "menu":
                self.run_menu()
            elif self.game_state == "playing":
                self.run_game()
            elif self.game_state == "game_over":
                self.run_game_over()
        
        pygame.quit()
        sys.exit()

    def run_menu(self):
        self.env.reset()
        self.winner = None
        self.game_state = "playing"

    def run_game(self):
        human_turn = self.env.turn == 0 # Assuming 0 is human, 1 is AI from env
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN and human_turn:
                for action, button in self.buttons.items():
                    if button["rect"].collidepoint(event.pos):
                        self.handle_human_move(action)
                        break

        if not human_turn and self.game_state == "playing":
            pygame.time.wait(500)
            self.handle_ai_move()

        self.screen.fill(BACKGROUND_COLOR)
        self.draw_stones()
        self.draw_buttons()
        self.draw_turn_indicator()
        pygame.display.flip()
        self.clock.tick(30)

    def handle_human_move(self, stones_to_take):
        if stones_to_take <= self.env.stones_remaining:
            _, _, terminated, _, _ = self.env.step(stones_to_take -1) # action is 0,1,2
            if terminated:
                self.winner = "Human" if self.env.stones_remaining == 0 else "AI"
                self.game_state = "game_over"

    def handle_ai_move(self):
        if self.q_table is not None:
            state = self.env.stones_remaining
            action_idx = np.argmax(self.q_table[state, :])
        else:
            # Fallback to the same optimal logic as the environment's opponent
            action_idx = (self.env.stones_remaining % 4) -1
            if action_idx < 0:
                action_idx = random.randint(0, 2)
        
        _, _, terminated, _, _ = self.env.step(action_idx)
        if terminated:
            self.winner = "AI" if self.env.stones_remaining == 0 else "Human"
            self.game_state = "game_over"


    def run_game_over(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.game_state = "menu"

        self.screen.fill(BACKGROUND_COLOR)
        winner_text = f"The {self.winner} wins!"
        text_surface = self.font.render(winner_text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(text_surface, text_rect)
        
        prompt_text = "Click to play again"
        prompt_surface = self.font.render(prompt_text, True, TEXT_COLOR)
        prompt_rect = prompt_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        self.screen.blit(prompt_surface, prompt_rect)

        pygame.display.flip()
        self.clock.tick(30)

    def draw_stones(self):
        stones_remaining = self.env.stones_remaining
        text_surface = self.font.render(f"Stones Remaining: {stones_remaining}", True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, 50))
        self.screen.blit(text_surface, text_rect)

        # Simple stone visualization
        stone_radius = 15
        stone_padding = 10
        total_width = stones_remaining * (2 * stone_radius + stone_padding) - stone_padding
        start_x = (SCREEN_WIDTH - total_width) // 2

        for i in range(stones_remaining):
            x = start_x + i * (2 * stone_radius + stone_padding)
            pygame.draw.circle(self.screen, STONE_COLOR, (x + stone_radius, 120), stone_radius)

    def draw_buttons(self):
        mouse_pos = pygame.mouse.get_pos()
        human_turn = self.env.turn == 0
        for action, button in self.buttons.items():
            rect = button["rect"]
            color = BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) and human_turn else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            
            text_surf = self.font.render(button["text"], True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)

    def draw_turn_indicator(self):
        turn_text = f"Turn: {'Human' if self.env.turn == 0 else 'AI'}"
        text_surface = self.font.render(turn_text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
        self.screen.blit(text_surface, text_rect)

def find_latest_q_table():
    script_dir = os.path.dirname(__file__)
    runs_dir = os.path.join(script_dir, 'runs')
    run_dirs = [d for d in glob.glob(os.path.join(runs_dir, "StoneGame__train__*")) if os.path.isdir(d)]
    if not run_dirs:
        return None
    latest_dir = max(run_dirs, key=os.path.getmtime)
    q_table_path = os.path.join(latest_dir, "q_table.npy")
    return q_table_path if os.path.exists(q_table_path) else None

if __name__ == "__main__":
    q_path = find_latest_q_table()
    if not q_path:
        print("No Q-table found. The AI will use a simple strategy.")
    engine = GameEngine(q_table_path=q_path)
    engine.run()
