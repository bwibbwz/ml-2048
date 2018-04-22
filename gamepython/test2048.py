from puzzle import GameGrid
from random import *
from logic import game_state
import time

# Event for manual inp
class event_rn:
    def __init__(self):
        self.char = ''

class test2048:
    def __init__(self, manual_input = True, random=True, steps=10, sleep=0):
        self.gamegrid = GameGrid(manual_input=manual_input)
        self.random = random
        self.steps = steps
        self.sleep = sleep

    def run(self, input_value=None):
        if self.random:
            self.run_random()
 
    def run_random(self):
        for k in range(self.steps):
            num = randint(0,3)
            event_rn.char = chr(num)
            self.gamegrid.key_down(event_rn)
            if game_state(self.gamegrid.matrix) == 'win' \
            or game_state(self.gamegrid.matrix) == 'lose':
                #time.sleep(1)
                return
            time.sleep(self.sleep)

    def get_status(self):
        return self.gamegrid.matrix

    def take_step(self, inp):
        event_rn.char = chr(inp)
        self.gamegrid.key_down(event_rn)
