from test2048 import test2048
import sys

man_in = bool(sys.argv[1])
rnd = bool(sys.argv[2])
stps = int(sys.argv[3])
slp = int(sys.argv[4])

tst = test2048(manual_input=man_in, random=rnd, steps=stps, sleep=slp)
tst.run()
