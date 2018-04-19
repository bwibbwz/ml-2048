from puzzle import GameGrid
from random import *

gamegrid = GameGrid(manual_input = True)
gamegrid.mainloop()

for k in range(10):
    num = randint(0,3)
    print(num)



