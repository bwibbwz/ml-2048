from test2048 import test2048
import sys

#man_in = bool(sys.argv[1])
#rnd = bool(sys.argv[2])
#stps = int(sys.argv[3])
#slp = int(sys.argv[4])

maxval = []

for i in range(1000):
    tst = test2048(manual_input=True, random=True, steps=10000, sleep=0)
    tst.run()
    maxval.append(max(max(tst.gamegrid.matrix)))
    tst.gamegrid.destroy()

print max(maxval)
