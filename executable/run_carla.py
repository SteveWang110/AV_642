import os

command = "py -3.8 ../../carla_lane_keeping_d3qn.py --version OTv1 --operation new --reward-function 5 --map Town04 --epsilon-decrement 0.005 --num-episodes 600 --max-steps 300 --random-spawn False"
os.system(command)