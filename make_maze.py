from __future__ import absolute_import, division, print_function

import curses
import os
import sys

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
from pycolab import ascii_art, human_ui
from pycolab.prefab_parts import drapes as prefab_drapes
from pycolab.prefab_parts import sprites as prefab_sprites

from src.maze import Maze
from src.maze_manager import MazeManager

n = 10
m = 10
method = 'dfs_backtrack'
manager = MazeManager()
maze = Maze(n, m)
foot_path = maze.generation_path
print(foot_path)
Maze =  ['#' * (m*2+1)] * (n*2+1)
init_step = foot_path[0]
Maze[init_step[0]+1] = Maze[init_step[0]+1][:init_step[1]+1]+'@'+Maze[init_step[0]+1][init_step[1]+2:]
for i in range(len(foot_path)-1):
    step = [foot_path[i+1][0]*2,foot_path[i+1][1]*2]
    Maze[step[0]+1] = Maze[step[0]+1][:step[1]+1]+'@'+Maze[step[0]+1][step[1]+2:]
    m_step= [int((init_step[0]+step[0])/2),int((init_step[1]+step[1])/2)]
    Maze[m_step[0]+1] = Maze[m_step[0]+1][:m_step[1]+1]+'@'+Maze[m_step[0]+1][m_step[1]+2:]
    init_step = step
Maze[0] = Maze[0][:0]+'+'+Maze[0][1:]
Maze[1] = Maze[1][:1]+'P'+Maze[1][2:]

# # visual the maze
# maze_using_btree = manager.add_existing_maze(maze)
# manager.show_maze(maze_using_btree.id)

MAZES_ART = [Maze]
MAZES_WHAT_LIES_BENEATH = '#'
STAR_ART = [' .       ',
            '        .',
            '       . ',
            '  .      ',
            '         ',
            '         ',
            '         ',
            '         ',]
COLOUR_FG = {' ': (0, 0, 0),        # Inky blackness of SPAAAACE
             '.': (949, 929, 999),  # These stars are full of lithium
             '@': (999, 862, 110),  # Shimmering golden coins
             '#': (764, 0, 999),    # Walls of the SPACE MAZE
             'P': (0, 999, 999),    # This is you, the player
            }  # Patroller C
COLOUR_BG = {'.': (0, 0, 0),        # Around the stars, inky blackness etc.
             '@': (0, 0, 0)}


def make_game(level):
  """Builds and returns a Scrolly Maze game for the selected level."""
  # A helper object that helps us with Scrolly-related setup paperwork.
  scrolly_info = prefab_drapes.Scrolly.PatternInfo(
      MAZES_ART[level], STAR_ART,
      board_northwest_corner_mark='+',
      what_lies_beneath=MAZES_WHAT_LIES_BENEATH[level])

  walls_kwargs = scrolly_info.kwargs('#')
  coins_kwargs = scrolly_info.kwargs('@')
  player_position = scrolly_info.virtual_position('P')

  return ascii_art.ascii_art_to_game(
      STAR_ART, what_lies_beneath=' ',
      sprites={
          'P': ascii_art.Partial(PlayerSprite, player_position)},
      drapes={
          '#': ascii_art.Partial(MazeDrape, **walls_kwargs),
          '@': ascii_art.Partial(CashDrape, **coins_kwargs)},
      # The base Backdrop class will do for a backdrop that just sits there.
      # In accordance with best practices, the one egocentric MazeWalker (the
      # player) is in a separate and later update group from all of the
      # pycolab entities that control non-traversable characters.
      update_schedule=[['#'], ['P'], ['@']],
      z_order='@#P')


class PlayerSprite(prefab_sprites.MazeWalker):
  """阻止玩家行走在障碍
  A `Sprite` for our player, the maze explorer.

  This egocentric `Sprite` requires no logic beyond tying actions to
  `MazeWalker` motion action helper methods, which keep the player from walking
  on top of obstacles.
  """

  def __init__(self, corner, position, character, virtual_position):
    """Constructor: player is egocentric and can't walk through walls."""
    super(PlayerSprite, self).__init__(
        corner, position, character, egocentric_scroller=True, impassable='#')
    self._teleport(virtual_position)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, things, layers  # Unused

    if actions == 0:    # go upward?
      self._north(board, the_plot)
    elif actions == 1:  # go downward?
      self._south(board, the_plot)
    elif actions == 2:  # go leftward?
      self._west(board, the_plot)
    elif actions == 3:  # go rightward?
      self._east(board, the_plot)
    elif actions == 4:  # do nothing?
      self._stay(board, the_plot)


class PatrollerSprite(prefab_sprites.MazeWalker):
  """Wanders back and forth horizontally, killing the player on contact."""

  def __init__(self, corner, position, character, virtual_position):
    """Constructor: changes virtual position to match the argument."""
    super(PatrollerSprite, self).__init__(corner, position, character, '#')
    self._teleport(virtual_position)
    # Choose our initial direction based on our character value.
    self._moving_east = bool(ord(character) % 2)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, layers, backdrop  # Unused.

    # We only move once every two game iterations.
    if the_plot.frame % 2:
      self._stay(board, the_plot)
      return

    # MazeWalker would make certain that we don't run into a wall, but only
    # if the sprite and the wall are visible on the game board. So, we have to
    # look after this ourselves in the general case.
    pattern_row, pattern_col = things['#'].pattern_position_prescroll(
        self.virtual_position, the_plot)
    next_to_wall = things['#'].whole_pattern[
        pattern_row, pattern_col+(1 if self._moving_east else -1)]
    if next_to_wall: self._moving_east = not self._moving_east

    # Make our move. If we're now in the same cell as the player, it's instant
    # game over!
    (self._east if self._moving_east else self._west)(board, the_plot)
    if self.virtual_position == things['P'].virtual_position:
      the_plot.terminate_episode()


class MazeDrape(prefab_drapes.Scrolly):
  """A scrolling `Drape` handling the maze scenery.

  This `Drape` requires no logic beyond tying actions to `Scrolly` motion
  action helper methods. Our job as programmers is to make certain that the
  actions we use have the same meaning between all `Sprite`s and `Drape`s in
  the same scrolling group (see `protocols/scrolling.py`).
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, things, layers  # Unused

    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions == 4:  # is the player doing nothing?
      self._stay(the_plot)


class CashDrape(prefab_drapes.Scrolly):
  """A scrolling `Drape` handling all of the coins.

  This `Drape` ties actions to `Scrolly` motion action helper methods, and once
  again we take care to map the same actions to the same methods. A little
  extra logic updates the scrolling pattern for when the player touches the
  coin, credits reward, and handles game termination.
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # If the player has reached a coin, credit one reward and remove the coin
    # from the scrolling pattern. If the player has obtained all coins, quit!
    player_pattern_position = self.pattern_position_prescroll(
        things['P'].position, the_plot)

    if self.whole_pattern[player_pattern_position]:
      the_plot.log('Coin collected at {}!'.format(player_pattern_position))
      the_plot.add_reward(100)
      self.whole_pattern[player_pattern_position] = False
      if not self.whole_pattern.any(): the_plot.terminate_episode()

    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions == 4:  # is the player doing nothing?
      self._stay(the_plot)
    elif actions == 5:  # does the player want to quit?
      the_plot.terminate_episode()


def main(argv=()):
  # Build a Scrolly Maze game.
  game = make_game(int(argv[1]) if len(argv) > 1 else 0)

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4,
                       'q': 5, 'Q': 5},
      delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)



