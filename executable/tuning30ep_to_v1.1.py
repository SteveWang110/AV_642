import os

command = "py -3.8 ../../carla_lane_keeping_d3qn.py --version 1.1 --operation tune --reward-function 5 --map Town04 --epsilon-decrement 0.005 --num-episodes 600 --max-steps 300 --random-spawn False --save-path OTv1_best_dqn_network_nn_model.pth"
os.system(command)