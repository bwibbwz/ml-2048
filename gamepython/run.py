from puzzle import GameGrid
from random import *
from logic import game_state
import time

# Event for manual inp
class event_rn:
    def __init__(self):
        self.char = ''

# Frame of GameGrid not an object(?)
# link matrix to gamegrid matrix

class run2048:
    def __init__(self, manual_input = True, random=True, steps=10, sleep=0):
        self.gamegrid = GameGrid(manual_input)
        self.random = random
        self.steps = steps
        self.sleep = sleep

        self.gamegrid.win_status = False

        self.step = 0
        self.old_matrix = []
        self.check_value = 0

    def run(self, input_value=None):
        self.old_matrix = self.gamegrid.matrix
        # if game_state(self.gamegrid.matrix) == 'lose':
        #
        if self.random and input_value is None:
            input_value = randint(0,3)

        assert input_value in range(4)

        event_rn.char = chr(input_value)
        self.take_step(event_rn)
        time.sleep(self.sleep)
        self.step += 1

    #def run_random(self):
    #    for k in range(self.steps):
    #        num = randint(0,3)
    #        event_rn.char = chr(num)
    #        self.take_step(event_rn)
    #        if game_state(self.gamegrid.matrix) == 'lose':
    #            return
    #        self.step += 1
    #        time.sleep(self.sleep)

    def get_status(self):
        # Need to figure out 'lose' state from check_matrix
        return self.gamegrid.matrix, int(self.check_matrix())

    def take_step(self, inp):
        self.gamegrid.key_down(inp)

    def check_matrix(self):
        if self.old_matrix == self.gamegrid.matrix:
            self.check_value += 1
        else:
            self.check_value = 0
        #return self.old_matrix == self.gamegrid.matrix
        return self.check_value
