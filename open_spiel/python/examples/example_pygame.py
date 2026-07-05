# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pygame viewer for OpenSpiel games (default: python_qwixx).

Controls:
  N: start a new game
  SPACE: apply one random legal action
  A: toggle autoplay
  ESC / window close: quit
"""

import random
import sys
from pathlib import Path

# Prefer the local repository checkout over any installed package so newly
# added Python games (e.g., python_qwixx) are visible.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from absl import app
from absl import flags

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel

try:
  import pygame
except ImportError as exc:
  raise ImportError(
      "pygame is required for example_pygame.py. Install with: pip install pygame"
  ) from exc

FLAGS = flags.FLAGS
flags.DEFINE_string("game_string", "python_qwixx", "Game string to load.")
flags.DEFINE_integer("width", 1280, "Window width.")
flags.DEFINE_integer("height", 800, "Window height.")
flags.DEFINE_integer("fps", 30, "Render FPS.")
flags.DEFINE_bool("human_player0", True, "If true, player 0 actions are selected by keyboard.")


_BG = (233, 225, 206)
_INK = (34, 33, 32)
_PANEL = (250, 246, 235)
_PANEL_BORDER = (188, 176, 152)
_ROW_COLORS = (
    (209, 63, 63),   # red
    (235, 180, 49),  # yellow
    (74, 163, 83),   # green
    (61, 104, 195),  # blue
)


def _row_numbers(row):
  """Returns the printed values for a row from left-to-right."""
  if row in (0, 1):
    return list(range(2, 13))
  return list(range(12, 1, -1))


def _is_qwixx_state(state):
  """Checks whether this state looks like the custom Python Qwixx state."""
  return all(
      hasattr(state, attr)
      for attr in ("_marks", "_row_locked", "_penalties", "_num_players", "_phase")
  )


def _random_action(state):
  """Samples one legal action for the current node type."""
  if state.is_chance_node():
    outcomes = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes)
    # random.choices is in stdlib and supports weighted sampling.
    return random.choices(action_list, weights=prob_list, k=1)[0]
  legal = state.legal_actions(state.current_player())
  if not legal:
    return None
  return random.choice(legal)


def _apply_action_and_track(state, action, step_count, last_decision_action):
  """Applies one action and updates step count plus the last decision label."""
  if action is None:
    return step_count, last_decision_action
  if not state.is_chance_node():
    last_decision_action = f"P{state.current_player()}: {state.action_to_string(state.current_player(), action)}"
  state.apply_action(action)
  return step_count + 1, last_decision_action


def _advance_until_decision_or_terminal(state, step_count, last_decision_action):
  """Automatically resolves chance nodes, then random non-human decisions.

  The loop stops when:
  - the game ends, or
  - a human-controlled player 0 decision node is reached.
  """
  while not state.is_terminal():
    if state.is_chance_node():
      action = _random_action(state)
      step_count, last_decision_action = _apply_action_and_track(
          state, action, step_count, last_decision_action
      )
      continue

    if _is_human_turn(state):
      return step_count, last_decision_action

    action = _random_action(state)
    step_count, last_decision_action = _apply_action_and_track(
        state, action, step_count, last_decision_action
    )

  return step_count, last_decision_action


def _draw_text_block(surface, font, x, y, text, color=(230, 230, 230), line_gap=6):
  """Draws a multiline text block and returns the new y cursor."""
  for line in text.splitlines():
    rendered = font.render(line, True, color)
    surface.blit(rendered, (x, y))
    y += rendered.get_height() + line_gap
  return y


def _state_header(state):
  if state.is_terminal():
    return "TERMINAL"
  if state.is_chance_node():
    return "CHANCE NODE"
  return f"DECISION NODE (player {state.current_player()})"


def _is_human_turn(state):
  """True when player 0 should act manually."""
  return (
      FLAGS.human_player0
      and not state.is_terminal()
      and not state.is_chance_node()
      and state.current_player() == 0
  )


def _draw_die(surface, rect, value, fill=(255, 255, 255), pip=(32, 32, 32)):
  """Draws a simple die tile with centered numeric value."""
  pygame.draw.rect(surface, fill, rect, border_radius=8)
  pygame.draw.rect(surface, (90, 90, 90), rect, width=2, border_radius=8)
  font = pygame.font.SysFont("consolas", max(16, rect.height // 2), bold=True)
  txt = font.render(str(value) if value is not None else "?", True, pip)
  surface.blit(txt, txt.get_rect(center=rect.center))


def _draw_qwixx_card(surface, state, player, rect, fonts):
  """Draws one player's scoresheet in a boardgame-style card layout."""
  title_font, body_font, tiny_font = fonts
  pygame.draw.rect(surface, _PANEL, rect, border_radius=12)

  active = (not state.is_terminal() and not state.is_chance_node() and state.current_player() == player)
  border_color = (70, 120, 220) if active else _PANEL_BORDER
  pygame.draw.rect(surface, border_color, rect, width=3, border_radius=12)

  pad = 12
  x = rect.x + pad
  y = rect.y + pad
  score = int(state._score(player))
  title = title_font.render(f"Player {player}   Score: {score}", True, _INK)
  surface.blit(title, (x, y))
  y += title.get_height() + 8

  # Penalty track.
  pen_label = body_font.render("Penalties:", True, _INK)
  surface.blit(pen_label, (x, y + 3))
  pen_x = x + pen_label.get_width() + 10
  box = 22
  penalties = int(state._penalties[player])
  for i in range(4):
    r = pygame.Rect(pen_x + i * (box + 6), y, box, box)
    pygame.draw.rect(surface, (247, 243, 230), r, border_radius=4)
    pygame.draw.rect(surface, (120, 112, 96), r, width=2, border_radius=4)
    if i < penalties:
      pygame.draw.line(surface, (180, 58, 58), (r.left + 4, r.top + 4), (r.right - 4, r.bottom - 4), 3)
      pygame.draw.line(surface, (180, 58, 58), (r.right - 4, r.top + 4), (r.left + 4, r.bottom - 4), 3)
  y += box + 10

  # Four colored rows.
  usable_w = rect.width - (2 * pad)
  row_h = 46
  cell_gap = 4
  stripe_w = 44
  cell_w = int((usable_w - stripe_w - 10 * cell_gap) / 11)

  for row in range(4):
    row_rect = pygame.Rect(x, y, usable_w, row_h)
    pygame.draw.rect(surface, (245, 241, 229), row_rect, border_radius=6)
    pygame.draw.rect(surface, (186, 177, 156), row_rect, width=1, border_radius=6)

    stripe = pygame.Rect(row_rect.x, row_rect.y, stripe_w, row_h)
    pygame.draw.rect(surface, _ROW_COLORS[row], stripe, border_radius=6)
    row_name = tiny_font.render(("R", "Y", "G", "B")[row], True, (255, 255, 255))
    surface.blit(row_name, row_name.get_rect(center=stripe.center))

    numbers = _row_numbers(row)
    for i, number in enumerate(numbers):
      cx = stripe.right + cell_gap + i * (cell_w + cell_gap)
      cr = pygame.Rect(cx, row_rect.y + 5, cell_w, row_h - 10)

      marked = bool(state._marks[player, row, i])
      lock_cell = (i == 10)
      if marked:
        pygame.draw.rect(surface, (222, 230, 214), cr, border_radius=4)
      else:
        pygame.draw.rect(surface, (255, 255, 250), cr, border_radius=4)

      border = _ROW_COLORS[row] if not lock_cell else (140, 113, 58)
      pygame.draw.rect(surface, border, cr, width=2, border_radius=4)

      n = tiny_font.render(str(number), True, _INK)
      surface.blit(n, n.get_rect(center=(cr.centerx, cr.centery - 8)))

      if lock_cell:
        lock_text = tiny_font.render("L", True, (93, 76, 38))
        surface.blit(lock_text, lock_text.get_rect(center=(cr.centerx, cr.centery + 8)))

      if marked:
        pygame.draw.line(surface, (20, 20, 20), (cr.left + 3, cr.top + 3), (cr.right - 3, cr.bottom - 3), 3)
        pygame.draw.line(surface, (20, 20, 20), (cr.right - 3, cr.top + 3), (cr.left + 3, cr.bottom - 3), 3)

      if lock_cell and state._row_locked[row]:
        pygame.draw.rect(surface, (201, 157, 54), cr, width=3, border_radius=4)

    y += row_h + 8


def _draw_qwixx_ui(
    surface,
    state,
    step_count,
    autoplay,
    last_decision_action,
    selected_human_action_index,
    fonts,
):
  """Renders a Qwixx-themed board UI using internal state fields."""
  title_font, body_font, tiny_font = fonts
  width, height = surface.get_size()

  surface.fill(_BG)

  # Left info panel.
  panel = pygame.Rect(14, 14, 320, height - 28)
  pygame.draw.rect(surface, _PANEL, panel, border_radius=12)
  pygame.draw.rect(surface, _PANEL_BORDER, panel, width=2, border_radius=12)

  y = panel.y + 12
  y = _draw_text_block(surface, title_font, panel.x + 12, y, "QWIXX Viewer", color=_INK, line_gap=2)
  y += 4
  mode = _state_header(state)
  y = _draw_text_block(
      surface,
      body_font,
      panel.x + 12,
      y,
      f"Game: {FLAGS.game_string}\nMode: {mode}\nStep: {step_count}\nAutoplay: {'ON' if autoplay else 'OFF'}",
      color=_INK,
      line_gap=4,
  )

  y += 8
  last_line = last_decision_action if last_decision_action else "(none yet)"
  y = _draw_text_block(surface, tiny_font, panel.x + 12, y, f"Last decision:\n{last_line}", color=(72, 64, 52), line_gap=2)

  # Dice strip.
  y += 8
  _draw_text_block(surface, body_font, panel.x + 12, y, "Dice", color=_INK, line_gap=0)
  y += 26
  dice_values = [
      state._white_1,
      state._white_2,
      state._red_die,
      state._yellow_die,
      state._green_die,
      state._blue_die,
  ]
  dice_fills = [
      (255, 255, 255),
      (255, 255, 255),
      _ROW_COLORS[0],
      _ROW_COLORS[1],
      _ROW_COLORS[2],
      _ROW_COLORS[3],
  ]
  dx = panel.x + 12
  for i, val in enumerate(dice_values):
    r = pygame.Rect(dx + i * 48, y, 40, 40)
    pip_color = (255, 255, 255) if i >= 2 else (22, 22, 22)
    _draw_die(surface, r, val, fill=dice_fills[i], pip=pip_color)

  y += 56
  if state.is_terminal():
    _draw_text_block(surface, body_font, panel.x + 12, y, f"Returns: {state.returns()}", color=(110, 82, 24), line_gap=2)
  else:
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      txt = "Chance outcomes:\n" + "\n".join(f"{a}: p={p:.3f}" for a, p in outcomes)
      _draw_text_block(surface, tiny_font, panel.x + 12, y, txt, color=(37, 84, 44), line_gap=2)
    else:
      cp = state.current_player()
      legal = state.legal_actions(cp)
      if _is_human_turn(state):
        lines = []
        start = max(0, selected_human_action_index - 4)
        end = min(len(legal), start + 9)
        for idx in range(start, end):
          marker = ">" if idx == selected_human_action_index else " "
          lines.append(f"{marker} [{idx}] {state.action_to_string(cp, legal[idx])}")
        txt = (
            f"Current player: {cp} (HUMAN)\n"
            f"Legal actions ({len(legal)}):\n"
            + "\n".join(lines)
            + "\nUse UP/DOWN + ENTER"
        )
      else:
        preview = "\n".join(state.action_to_string(cp, a) for a in legal[:10])
        if len(legal) > 10:
          preview += f"\n... ({len(legal) - 10} more)"
        txt = f"Current player: {cp}\nLegal actions ({len(legal)}):\n{preview}"
      _draw_text_block(surface, tiny_font, panel.x + 12, y, txt, color=(95, 61, 27), line_gap=2)

  controls = "N=new  SPACE=step  A=autoplay  UP/DOWN+ENTER human  ESC=quit"
  _draw_text_block(surface, tiny_font, panel.x + 12, panel.bottom - 24, controls, color=(98, 90, 75), line_gap=0)

  # Card area (one score sheet per player).
  cards_area = pygame.Rect(panel.right + 12, 14, width - panel.width - 40, height - 28)
  num_players = int(state._num_players)
  cols = 1 if num_players == 1 else 2
  rows = (num_players + cols - 1) // cols
  gap = 10
  card_w = (cards_area.width - gap * (cols - 1)) // cols
  card_h = (cards_area.height - gap * (rows - 1)) // rows

  for player in range(num_players):
    row = player // cols
    col = player % cols
    rect = pygame.Rect(
        cards_area.x + col * (card_w + gap),
        cards_area.y + row * (card_h + gap),
        card_w,
        card_h,
    )
    _draw_qwixx_card(surface, state, player, rect, fonts)


def main(argv):
  del argv

  pygame.init()
  pygame.display.set_caption("OpenSpiel Pygame Example")
  screen = pygame.display.set_mode((FLAGS.width, FLAGS.height))
  clock = pygame.time.Clock()
  font = pygame.font.SysFont("consolas", 24, bold=True)
  body_font = pygame.font.SysFont("consolas", 18)
  small_font = pygame.font.SysFont("consolas", 14)

  game = pyspiel.load_game(FLAGS.game_string)
  state = game.new_initial_state()
  autoplay = False
  step_count = 0
  last_decision_action = ""
  selected_human_action_index = 0

  step_count, last_decision_action = _advance_until_decision_or_terminal(
      state, step_count, last_decision_action
  )

  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          running = False
        elif event.key == pygame.K_n:
          state = game.new_initial_state()
          step_count = 0
          last_decision_action = ""
          selected_human_action_index = 0
          step_count, last_decision_action = _advance_until_decision_or_terminal(
              state, step_count, last_decision_action
          )
        elif event.key == pygame.K_a:
          autoplay = not autoplay
        elif event.key == pygame.K_UP and _is_human_turn(state):
          legal = state.legal_actions(state.current_player())
          if legal:
            selected_human_action_index = max(0, selected_human_action_index - 1)
        elif event.key == pygame.K_DOWN and _is_human_turn(state):
          legal = state.legal_actions(state.current_player())
          if legal:
            selected_human_action_index = min(len(legal) - 1, selected_human_action_index + 1)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and _is_human_turn(state):
          legal = state.legal_actions(state.current_player())
          if legal:
            selected_human_action_index = max(0, min(selected_human_action_index, len(legal) - 1))
            action = legal[selected_human_action_index]
            step_count, last_decision_action = _apply_action_and_track(
                state, action, step_count, last_decision_action
            )
            step_count, last_decision_action = _advance_until_decision_or_terminal(
                state, step_count, last_decision_action
            )
        elif event.key == pygame.K_SPACE and not state.is_terminal():
          # SPACE advances the game until the next decision node or terminal.
          if _is_human_turn(state):
            legal = state.legal_actions(state.current_player())
            if legal:
              selected_human_action_index = max(0, min(selected_human_action_index, len(legal) - 1))
              action = legal[selected_human_action_index]
              step_count, last_decision_action = _apply_action_and_track(
                  state, action, step_count, last_decision_action
              )
          else:
            step_count, last_decision_action = _advance_until_decision_or_terminal(
                state, step_count, last_decision_action
            )

    if autoplay and not state.is_terminal():
      # In autoplay mode, keep human control for player 0 decision nodes.
      if not _is_human_turn(state):
        step_count, last_decision_action = _advance_until_decision_or_terminal(
            state, step_count, last_decision_action
        )

    # Clamp selection when legal action count changes.
    if _is_human_turn(state):
      legal = state.legal_actions(state.current_player())
      if legal:
        selected_human_action_index = max(0, min(selected_human_action_index, len(legal) - 1))
      else:
        selected_human_action_index = 0

    if _is_qwixx_state(state):
      _draw_qwixx_ui(
          screen,
          state,
          step_count,
          autoplay,
          last_decision_action,
          selected_human_action_index,
          (font, body_font, small_font),
      )
    else:
      screen.fill((20, 22, 26))
      y = 16
      y = _draw_text_block(
          screen,
          body_font,
          16,
          y,
          (
              f"Game: {FLAGS.game_string}\n"
              f"Mode: {_state_header(state)}\n"
              f"Step: {step_count}\n"
              f"Autoplay: {'ON' if autoplay else 'OFF'}"
          ),
          color=(240, 240, 240),
      )
      y += 8
      y = _draw_text_block(screen, body_font, 16, y, "State:")
      y = _draw_text_block(screen, small_font, 16, y, str(state), color=(200, 220, 240), line_gap=4)
      controls = "Controls: N=new, SPACE=step random, A=autoplay, ESC=quit"
      _draw_text_block(screen, small_font, 16, FLAGS.height - 28, controls, color=(170, 170, 170), line_gap=0)

    pygame.display.flip()
    clock.tick(max(1, FLAGS.fps))

  pygame.quit()
  return 0


if __name__ == "__main__":
  app.run(main)