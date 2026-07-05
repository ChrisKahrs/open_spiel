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

# Lint as python3
"""Qwixx implemented in Python.

This is a turn-based approximation of Qwixx with explicit chance nodes.
Each turn rolls 6 dice, then all players get one white-sum decision in order,
then the active player gets one colored-die decision.
"""

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

# Basic game-size constants.
# Qwixx supports multiple players, four colored rows, and 11 values per row.
_MIN_PLAYERS = 2
_MAX_PLAYERS = 5
_NUM_ROWS = 4
_ROW_LEN = 11

# Action encoding.
_PASS_ACTION = 0
_WHITE_ROW_OFFSET = 1  # [1, 4] -> red, yellow, green, blue.
_COLOR_ACTION_OFFSET = 5
_COLOR_VALUE_COUNT = 11  # Values 2..12.

# Row ids.
_RED = 0
_YELLOW = 1
_GREEN = 2
_BLUE = 3
_ROW_NAMES = ("red", "yellow", "green", "blue")

# Row value ordering in Qwixx.
# Red and yellow increase left-to-right, while green and blue decrease.
_ROW_VALUES = (
    (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    (12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2),
    (12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2),
)

_VALUE_TO_INDEX = [
    {value: i for i, value in enumerate(_ROW_VALUES[row])}
    for row in range(_NUM_ROWS)
]


class _Phase:
  """State-machine phases for one full Qwixx turn.

  A turn has six chance phases (dice rolls), then two decision phases:
  white-sum choices for each player and one colored-die choice for the
  active player.
  """

  ROLL_WHITE_1 = 0
  ROLL_WHITE_2 = 1
  ROLL_RED = 2
  ROLL_YELLOW = 3
  ROLL_GREEN = 4
  ROLL_BLUE = 5
  WHITE_DECISION = 6
  COLOR_DECISION = 7


_DEFAULT_PARAMS = {"players": 2}

# OpenSpiel metadata describing the game and what API surfaces it supports.
_GAME_TYPE = pyspiel.GameType(
    short_name="python_qwixx",
    long_name="Python Qwixx",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_MAX_PLAYERS,
    min_num_players=_MIN_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification=_DEFAULT_PARAMS,
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_COLOR_ACTION_OFFSET + _NUM_ROWS * _COLOR_VALUE_COUNT,
    max_chance_outcomes=6,
    num_players=_MIN_PLAYERS,
    min_utility=-20.0,
    max_utility=264.0,
    utility_sum=None,
    max_game_length=500,
)


def _encode_white_action(row):
  """Encodes a white-sum action for a specific row into one action id."""
  return _WHITE_ROW_OFFSET + row


def _decode_white_action(action):
  """Decodes a white-sum action id back into its row index."""
  return action - _WHITE_ROW_OFFSET


def _encode_color_action(row, value):
  """Encodes a colored-die mark (row, value) into one action id."""
  return _COLOR_ACTION_OFFSET + row * _COLOR_VALUE_COUNT + (value - 2)


def _decode_color_action(action):
  """Decodes a colored-die action id into `(row, value)`.

  Values are in the Qwixx numeric range 2..12.
  """
  rel = action - _COLOR_ACTION_OFFSET
  row = rel // _COLOR_VALUE_COUNT
  value = 2 + (rel % _COLOR_VALUE_COUNT)
  return row, value


class QwixxGame(pyspiel.Game):
  """A Python version of Qwixx."""

  def __init__(self, params=None):
    """Builds game metadata and validates user-specified parameters.

    Args:
      params: Optional parameter dictionary. Supports `players`.
    """
    params = params or dict(_DEFAULT_PARAMS)
    num_players = params.get("players", _DEFAULT_PARAMS["players"])
    if num_players < _MIN_PLAYERS or num_players > _MAX_PLAYERS:
      raise ValueError(
          f"players must be in [{_MIN_PLAYERS}, {_MAX_PLAYERS}], got {num_players}")

    game_info = pyspiel.GameInfo(
        num_distinct_actions=_GAME_INFO.num_distinct_actions,
        max_chance_outcomes=_GAME_INFO.max_chance_outcomes,
        num_players=num_players,
        min_utility=_GAME_INFO.min_utility,
        max_utility=_GAME_INFO.max_utility,
        utility_sum=None,
        max_game_length=_GAME_INFO.max_game_length,
    )
    super().__init__(_GAME_TYPE, game_info, params)

  def new_initial_state(self):
    """Creates the starting state for a new Qwixx game."""
    return QwixxState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Creates a standard OpenSpiel public-information observer."""
    return IIGObserverForPublicInfoGame(iig_obs_type, params)


class QwixxState(pyspiel.State):
  """A Python state implementation for Qwixx."""

  def __init__(self, game):
    """Initializes turn state, dice, player sheets, penalties, and locks."""
    super().__init__(game)
    self._num_players = game.num_players()
    self._phase = _Phase.ROLL_WHITE_1
    self._active_player = 0
    self._white_order_index = 0
    self._terminal = False

    # Dice values for the current turn.
    self._white_1 = None
    self._white_2 = None
    self._red_die = None
    self._yellow_die = None
    self._green_die = None
    self._blue_die = None

    # Mark sheets: [player, row, index-on-row].
    self._marks = np.zeros((self._num_players, _NUM_ROWS, _ROW_LEN), dtype=bool)
    self._row_locked = np.zeros(_NUM_ROWS, dtype=bool)
    self._penalties = np.zeros(self._num_players, dtype=np.int32)
    self._active_marked_this_turn = False

  def current_player(self):
    """Returns who acts now: TERMINAL, CHANCE, or a concrete player id.

    - During roll phases, chance acts.
    - During white decisions, each player acts in order.
    - During colored decision, only the active player acts.
    """
    if self._terminal:
      return pyspiel.PlayerId.TERMINAL
    if self._phase <= _Phase.ROLL_BLUE:
      return pyspiel.PlayerId.CHANCE
    if self._phase == _Phase.WHITE_DECISION:
      return (self._active_player + self._white_order_index) % self._num_players
    return self._active_player

  def _legal_actions(self, player):
    """Returns legal actions for the current phase.

    The `player` argument is ignored because legal moves are phase-dependent
    and driven by `current_player()` and turn context.
    """
    del player
    if self._phase == _Phase.WHITE_DECISION:
      return self._legal_white_actions(self.current_player())
    if self._phase == _Phase.COLOR_DECISION:
      return self._legal_color_actions(self._active_player)
    return []

  def chance_outcomes(self):
    """Returns equiprobable d6 outcomes for each die roll phase."""
    assert self.is_chance_node()
    return [(i, 1.0 / 6.0) for i in range(1, 7)]

  def _apply_action(self, action):
    """Applies chance or player actions and advances the turn state machine.

    Flow:
    1. Chance phases fill six dice values.
    2. White-decision phase lets every player optionally mark white sum.
    3. Colored-decision phase lets active player optionally mark one color sum.
    4. If active player marked nothing in both opportunities, apply penalty.
    5. Check terminal conditions, otherwise start next turn.
    """
    if self.is_chance_node():
      self._apply_chance(action)
      return

    if self._phase == _Phase.WHITE_DECISION:
      player = self.current_player()
      if action != _PASS_ACTION:
        row = _decode_white_action(action)
        if self._mark(player, row, self._white_sum()):
          if player == self._active_player:
            self._active_marked_this_turn = True
      self._white_order_index += 1
      if self._white_order_index >= self._num_players:
        self._phase = _Phase.COLOR_DECISION
      return

    # COLOR_DECISION
    if action != _PASS_ACTION:
      row, value = _decode_color_action(action)
      if self._mark(self._active_player, row, value):
        self._active_marked_this_turn = True

    if not self._active_marked_this_turn:
      self._penalties[self._active_player] += 1

    self._terminal = self._is_game_over()
    if not self._terminal:
      self._start_next_turn()

  def _action_to_string(self, player, action):
    """Builds readable action labels for logs and debugging."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"roll:{action}"
    if action == _PASS_ACTION:
      return "pass"
    if _WHITE_ROW_OFFSET <= action < _WHITE_ROW_OFFSET + _NUM_ROWS:
      row = _decode_white_action(action)
      return f"white:{_ROW_NAMES[row]}={self._white_sum()}"
    row, value = _decode_color_action(action)
    return f"color:{_ROW_NAMES[row]}={value}"

  def is_terminal(self):
    """Returns whether the game has reached a terminal state."""
    return self._terminal

  def returns(self):
    """Returns final per-player scores at terminal, else all zeros."""
    if not self._terminal:
      return [0.0] * self._num_players
    return [float(self._score(player)) for player in range(self._num_players)]

  def __str__(self):
    """Returns a compact multiline debug snapshot of the whole game state."""
    lines = [
        f"phase={self._phase} active={self._active_player} "
        f"white=({self._white_1},{self._white_2}) "
        f"color=({self._red_die},{self._yellow_die},{self._green_die},{self._blue_die})",
        f"locked={self._row_locked.tolist()} penalties={self._penalties.tolist()}",
    ]
    for player in range(self._num_players):
      row_parts = []
      for row in range(_NUM_ROWS):
        marks = "".join("x" if v else "." for v in self._marks[player, row])
        row_parts.append(f"{_ROW_NAMES[row][0].upper()}:{marks}")
      lines.append(f"P{player} " + " ".join(row_parts) + f" score={self._score(player)}")
    return "\n".join(lines)

  def _apply_chance(self, action):
    """Applies one die result and advances to the next roll/decision phase."""
    if self._phase == _Phase.ROLL_WHITE_1:
      self._white_1 = action
      self._phase = _Phase.ROLL_WHITE_2
    elif self._phase == _Phase.ROLL_WHITE_2:
      self._white_2 = action
      self._phase = _Phase.ROLL_RED
    elif self._phase == _Phase.ROLL_RED:
      self._red_die = action
      self._phase = _Phase.ROLL_YELLOW
    elif self._phase == _Phase.ROLL_YELLOW:
      self._yellow_die = action
      self._phase = _Phase.ROLL_GREEN
    elif self._phase == _Phase.ROLL_GREEN:
      self._green_die = action
      self._phase = _Phase.ROLL_BLUE
    else:
      self._blue_die = action
      self._phase = _Phase.WHITE_DECISION
      self._white_order_index = 0
      self._active_marked_this_turn = False

  def _start_next_turn(self):
    """Rotates active player and resets per-turn transient dice/flags."""
    self._active_player = (self._active_player + 1) % self._num_players
    self._phase = _Phase.ROLL_WHITE_1
    self._white_order_index = 0
    self._white_1 = None
    self._white_2 = None
    self._red_die = None
    self._yellow_die = None
    self._green_die = None
    self._blue_die = None
    self._active_marked_this_turn = False

  def _white_sum(self):
    """Returns the sum of the two white dice for this turn."""
    return self._white_1 + self._white_2

  def _legal_white_actions(self, player):
    """Returns legal white-sum actions for a given player.

    White actions are optional, so pass is always legal.
    """
    value = self._white_sum()
    actions = [_PASS_ACTION]
    for row in range(_NUM_ROWS):
      if self._can_mark(player, row, value):
        actions.append(_encode_white_action(row))
    return actions

  def _legal_color_actions(self, player):
    """Returns legal colored-die actions for the active player.

    The player may pass or mark one row/value derived from one white + one
    matching colored die.
    """
    actions = {_PASS_ACTION}
    color_dice = (self._red_die, self._yellow_die, self._green_die, self._blue_die)
    for row in range(_NUM_ROWS):
      if self._row_locked[row]:
        continue
      for white in (self._white_1, self._white_2):
        value = white + color_dice[row]
        if 2 <= value <= 12 and self._can_mark(player, row, value):
          actions.add(_encode_color_action(row, value))
    return sorted(actions)

  def _can_mark(self, player, row, value):
    """Checks whether a proposed mark is legal under Qwixx constraints.

    Constraints enforced:
    - Row is not globally locked.
    - Value exists in row domain.
    - Cell is not already marked by this player.
    - Marks move strictly left-to-right in that player's row.
    - Lock cell (right-most) requires at least 5 previous marks in row.
    """
    if self._row_locked[row] or value not in _VALUE_TO_INDEX[row]:
      return False

    idx = _VALUE_TO_INDEX[row][value]
    row_marks = self._marks[player, row]
    if row_marks[idx]:
      return False

    marked_indices = np.flatnonzero(row_marks)
    if marked_indices.size > 0 and idx <= int(marked_indices.max()):
      return False

    # Lock value is the right-most cell. It requires at least 5 earlier marks.
    if idx == (_ROW_LEN - 1) and int(np.sum(row_marks)) < 5:
      return False

    return True

  def _mark(self, player, row, value):
    """Applies a legal mark and locks the row if lock cell is marked.

    Returns:
      True if mark was applied, False if illegal.
    """
    if not self._can_mark(player, row, value):
      return False

    idx = _VALUE_TO_INDEX[row][value]
    self._marks[player, row, idx] = True
    if idx == (_ROW_LEN - 1):
      self._row_locked[row] = True
    return True

  def _score(self, player):
    """Computes final Qwixx score for one player.

    Row score is triangular number `n(n+1)/2` where `n` is marks in row.
    Each penalty subtracts 5 points.
    """
    total = 0
    for row in range(_NUM_ROWS):
      count = int(np.sum(self._marks[player, row]))
      total += count * (count + 1) // 2
    total -= 5 * int(self._penalties[player])
    return total

  def _is_game_over(self):
    """Checks Qwixx terminal conditions.

    Game ends when at least two rows are locked, or any player reaches
    four penalties.
    """
    return int(np.sum(self._row_locked)) >= 2 or int(np.max(self._penalties)) >= 4


# Register this Python game so `pyspiel.load_game("python_qwixx")` can create it.
pyspiel.register_game(_GAME_TYPE, QwixxGame)