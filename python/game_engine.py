
import pygame
import numpy as np
import sys
import os
import glob
import random
from game_env import StoneGameEnv
# --- Constants ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BACKGROUND_COLOR = (240, 240, 240)
STONE_COLOR = (100, 100, 100)
TEXT_COLOR = (20, 20, 20)
BUTTON_COLOR = (200, 200, 200)
BUTTON_HOVER_COLOR = (180, 180, 180)
FONT_SIZE = 36
INSIGHT_PANEL_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (30, 144, 255) # A nice blue color
AI_PANEL_WIDTH = 320
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
        self.turn = "human" # human or ai
        self.winner = None
        self.last_move = {"human": None, "ai": None}
        self.ai_think_start_time = 0
        self.ai_status_text = ""
        self.wins = {"human": 0, "ai": 0}
        # Button setup
        self.buttons = {}
        button_width, button_height = 180, 60
        button_y = SCREEN_HEIGHT - 120
        button_actions = ["Take 1", "Take 2", "Take 3"]
        total_button_width = len(button_actions) * button_width + (len(button_actions) - 1) * 30
        start_x = (SCREEN_WIDTH - AI_PANEL_WIDTH - total_button_width) // 2
        for i, text in enumerate(button_actions):
            rect = pygame.Rect(start_x + i * (button_width + 30), button_y, button_width, button_height)
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
        self.turn = "human"
        self.ai_status_text = "Waiting for player..."
        self.winner = None
        self.last_move = {"human": None, "ai": None}
        self.game_state = "playing"
    def run_game(self):
        is_ai_turn = self.turn == "ai" and self.game_state == "playing"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN and not is_ai_turn:
                for action, button in self.buttons.items():
                    if button["rect"].collidepoint(event.pos):
                        self.handle_human_move(action)
                        break
       
        # AI logic with delay
        if is_ai_turn:
            # If the AI just got the turn, record the start time
            if self.ai_think_start_time == 0:
                self.ai_think_start_time = pygame.time.get_ticks()
                self.ai_status_text = "Thinking..."
            # After a delay, make the move
            if pygame.time.get_ticks() - self.ai_think_start_time > 1000: # 1-second delay
                self.handle_ai_move()
                self.ai_think_start_time = 0 # Reset timer
        # Drawing
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_stones()
        self.draw_buttons()
        self.draw_turn_indicator()
        self.draw_last_move_info()
        self.draw_score_panel()
        self.draw_ai_panel()
        pygame.display.flip()
        self.clock.tick(30)
    def handle_human_move(self, stones_to_take):
        if stones_to_take <= self.env.stones_remaining:
            self.last_move["human"] = stones_to_take
            self.last_move["ai"] = None # Clear AI's last move
            self.env.stones_remaining -= stones_to_take
            if self.env.stones_remaining == 0:
                self.winner = "Human"
                self.wins["human"] += 1
                self.game_state = "game_over"
            else:
                self.turn = "ai"
    def handle_ai_move(self):
        state = self.env.stones_remaining
        if self.q_table is not None:
            # AI uses its learned Q-table
            # We must account for the fact the action in the table might be invalid
            valid_actions = [a for a in range(self.env.action_space.n) if (a + 1) <= state]
            if not valid_actions: # Should not happen if game logic is correct
                self.winner = "Human" # AI has no valid moves, human wins
                self.wins["human"] += 1
                self.game_state = "game_over"
                return
            # From the valid actions, choose the one with the highest Q-value
            q_values = self.q_table[state, valid_actions]
            best_action_idx = np.argmax(q_values)
            action_to_take = valid_actions[best_action_idx]
            stones_to_take = action_to_take + 1
           
        else:
            # Fallback to the same optimal logic as the environment's opponent
            stones_to_take = self.env.stones_remaining % 4
            if stones_to_take == 0:
                stones_to_take = random.randint(1, 3)
       
        stones_to_take = min(stones_to_take, self.env.stones_remaining)
        self.last_move["ai"] = stones_to_take
        self.ai_status_text = f"Took {stones_to_take} stone(s)."
        self.env.stones_remaining -= stones_to_take
        if self.env.stones_remaining == 0:
            self.winner = "AI"
            self.wins["ai"] += 1
            self.game_state = "game_over"
        else:
            self.turn = "human"
    def run_game_over(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.game_state = "menu"
        self.screen.fill(BACKGROUND_COLOR)
       
        game_center_x = (SCREEN_WIDTH - AI_PANEL_WIDTH) // 2
       
        if self.winner == "Human":
            art_lines = [
                "  o  ",
                " /|\\ ",
                " / \\ ",
                "HUMAN",
                "WINS!"
            ]
        else:  # AI wins
            art_lines = [
                " [ ] ",
                " /||\\",
                " // \\\\",
                "  AI ",
                "WINS!"
            ]
       
        line_height = FONT_SIZE
        total_art_height = len(art_lines) * line_height
        starting_y = (SCREEN_HEIGHT - total_art_height) // 2
       
        for line in art_lines:
            text_surface = self.font.render(line, True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(game_center_x, starting_y))
            self.screen.blit(text_surface, text_rect)
            starting_y += line_height
       
        prompt_text = "Click to play again"
        prompt_surface = self.font.render(prompt_text, True, TEXT_COLOR)
        prompt_rect = prompt_surface.get_rect(center=(game_center_x, starting_y + 50))
        self.screen.blit(prompt_surface, prompt_rect)
        self.draw_ai_panel()
        pygame.display.flip()
        self.clock.tick(30)
    def draw_stones(self):
        stones_remaining = self.env.stones_remaining
        text_surface = self.font.render(f"Stones Remaining: {stones_remaining}", True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=((SCREEN_WIDTH - AI_PANEL_WIDTH) // 2, 80))
        self.screen.blit(text_surface, text_rect)
        # Draw stones in a 3-row grid layout (7-7-7)
        stone_radius = 20
        stone_padding = 25
        stones_per_row = 7
       
        start_y = 150
        row_height = 2 * stone_radius + stone_padding
        for i in range(stones_remaining):
            row = i // stones_per_row
            col = i % stones_per_row
           
            # Center each row
            stones_in_this_row = min(stones_per_row, stones_remaining - row * stones_per_row if row == (stones_remaining -1)//stones_per_row else stones_per_row)
            row_width = stones_in_this_row * (2 * stone_radius + stone_padding) - stone_padding
            start_x = ((SCREEN_WIDTH - AI_PANEL_WIDTH) - row_width) // 2
            x = start_x + col * (2 * stone_radius + stone_padding)
            y = start_y + row * row_height
           
            pygame.draw.circle(self.screen, STONE_COLOR, (x + stone_radius, y + stone_radius), stone_radius)
    def draw_buttons(self):
        mouse_pos = pygame.mouse.get_pos()
        for action, button in self.buttons.items():
            # Adjust button position to be centered in the game area
            game_area_width = SCREEN_WIDTH - AI_PANEL_WIDTH
            total_button_width = len(self.buttons) * button["rect"].width + (len(self.buttons) - 1) * 30
            start_x = (game_area_width - total_button_width) // 2
           
            button_x = start_x + (action - 1) * (button["rect"].width + 30)
           
            rect = pygame.Rect(button_x, button["rect"].y, button["rect"].width, button["rect"].height)
            color = BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) and self.turn == "human" else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
           
            text_surf = self.font.render(button["text"], True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
    def draw_turn_indicator(self):
        turn_text = f"Turn: {self.turn.capitalize()}"
        text_surface = self.font.render(turn_text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=((SCREEN_WIDTH - AI_PANEL_WIDTH) // 2, SCREEN_HEIGHT - 140))
        self.screen.blit(text_surface, text_rect)
    def draw_last_move_info(self):
        # Persistent display for both player's last moves
        human_text = f"Your Last Move: Took {self.last_move['human']}" if self.last_move["human"] else "You are up!"
        ai_text = f"AI's Last Move: Took {self.last_move['ai']}" if self.last_move["ai"] else ""
        human_surf = self.font.render(human_text, True, TEXT_COLOR)
        ai_surf = self.font.render(ai_text, True, TEXT_COLOR)
       
        # Center this text above the buttons
        human_rect = human_surf.get_rect(center=((SCREEN_WIDTH - AI_PANEL_WIDTH) // 2, SCREEN_HEIGHT - 220))
        ai_rect = ai_surf.get_rect(center=((SCREEN_WIDTH - AI_PANEL_WIDTH) // 2, SCREEN_HEIGHT - 180))
       
        self.screen.blit(human_surf, human_rect)
        self.screen.blit(ai_surf, ai_rect)
    def draw_score_panel(self):
        score_text = f"Score: Human {self.wins['human']} - {self.wins['ai']} AI"
        text_surface = self.font.render(score_text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(topleft=(40, 40))
        self.screen.blit(text_surface, text_rect)
    def draw_ai_panel(self):
        panel_rect = pygame.Rect(SCREEN_WIDTH - AI_PANEL_WIDTH, 0, AI_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, INSIGHT_PANEL_COLOR, panel_rect)
        # Title
        title_font = pygame.font.Font(None, 40)
        title_surf = title_font.render("AI INSIGHTS", True, TEXT_COLOR)
        title_rect = title_surf.get_rect(center=(panel_rect.centerx, 50))
        self.screen.blit(title_surf, title_rect)
        # Status
        status_font = pygame.font.Font(None, 32)
        status_text = f"Status: {self.ai_status_text}"
        status_surf = status_font.render(status_text, True, TEXT_COLOR)
        status_rect = status_surf.get_rect(midleft=(panel_rect.left + 25, 120))
        self.screen.blit(status_surf, status_rect)
        # Q-values section
        q_title_font = pygame.font.Font(None, 32)
        q_title_surf = q_title_font.render("Q-Values (Current State):", True, TEXT_COLOR)
        q_title_rect = q_title_surf.get_rect(midleft=(panel_rect.left + 25, 200))
        self.screen.blit(q_title_surf, q_title_rect)
        if self.q_table is not None:
            state = self.env.stones_remaining
            q_values = self.q_table[state]
           
            best_action = -1
            if self.turn == "ai" and self.ai_think_start_time != 0:
                valid_actions = [a for a in range(self.env.action_space.n) if (a + 1) <= state]
                if valid_actions:
                    best_action_idx = np.argmax(q_values[valid_actions])
                    best_action = valid_actions[best_action_idx]
            for i, q_val in enumerate(q_values):
                action_num = i + 1
                is_best = (i == best_action)
                is_valid = action_num <= state
               
                text = f"Take {action_num}: {q_val:.3f}"
                color = TEXT_COLOR
                if not is_valid:
                    color = (180, 180, 180) # Gray out invalid moves
                elif is_best:
                    color = HIGHLIGHT_COLOR # Highlight the best valid move
               
                font = pygame.font.Font(None, 30)
                q_surf = font.render(text, True, color)
                q_rect = q_surf.get_rect(midleft=(panel_rect.left + 25, 250 + i * 35))
                self.screen.blit(q_surf, q_rect)
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
