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
"""Pure Python Tic-Tac-Toe variant registered as `tic_tac_toep`.

This mirrors the API/behavior of the C++ `tic_tac_toe` game while running
entirely in Python so it can be loaded via:

  pyspiel.load_game("tic_tac_toep")
"""

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2
_NUM_ROWS = 3
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(
    short_name="tic_tac_toep",
    long_name="Python Tic-Tac-Toe P",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CELLS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_NUM_CELLS)


class TicTacToepGame(pyspiel.Game):
  """A Python version of Tic-Tac-Toe registered as `tic_tac_toep`."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    return TicTacToepState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class TicTacToepState(pyspiel.State):
  """A Python state implementation for `tic_tac_toep`."""

  def __init__(self, game):
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board = np.full((_NUM_ROWS, _NUM_COLS), ".")

  def current_player(self):
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    del player
    return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]

  def _apply_action(self, action):
    self.board[_coord(action)] = "x" if self._cur_player == 0 else "o"
    if _line_exists(self.board):
      self._is_terminal = True
      self._player0_score = 1.0 if self._cur_player == 0 else -1.0
    elif all(self.board.ravel() != "."):
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def _action_to_string(self, player, action):
    row, col = _coord(action)
    return "{}({},{})".format("x" if player == 0 else "o", row, col)

  def is_terminal(self):
    return self._is_terminal

  def returns(self):
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    return _board_to_string(self.board)


class BoardObserver:
  """Observer, conforming to the PyObserver interface."""

  def __init__(self, params):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    shape = (1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    del player
    obs = self.dict["observation"]
    obs.fill(0)
    for row in range(_NUM_ROWS):
      for col in range(_NUM_COLS):
        cell_state = ".ox".index(state.board[row, col])
        obs[cell_state, row, col] = 1

  def string_from(self, state, player):
    del player
    return _board_to_string(state.board)


def _line_value(line):
  if all(line == "x") or all(line == "o"):
    return line[0]


def _line_exists(board):
  return (_line_value(board[0]) or _line_value(board[1]) or
          _line_value(board[2]) or _line_value(board[:, 0]) or
          _line_value(board[:, 1]) or _line_value(board[:, 2]) or
          _line_value(board.diagonal()) or
          _line_value(np.fliplr(board).diagonal()))


def _coord(move):
  return (move // _NUM_COLS, move % _NUM_COLS)


def _board_to_string(board):
  return "\n".join("".join(row) for row in board)


pyspiel.register_game(_GAME_TYPE, TicTacToepGame)