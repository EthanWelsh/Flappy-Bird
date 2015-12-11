Playing Flappy Bird using a reinforcement learning agent! The algorithm used is Q-Learning. Code for the GUI and basic animation and physics borrowed from [here.](https://github.com/TimoWilken/flappy-bird-pygame).
 
State Representation
--------------------
 - For my state representation, used the following:
    + *delta_x*: The horizontal distance from the bird to the rightmost edge of the pipe.
    + *delta_y*: The vertical distance from the bird to the gap in the pipes.
    + *is_jumping*: A boolean variable that expresses if the bird is jumping or falling.
    + *bird_lmh*: A value which describes which third of the screen the bird currently occupies
    + *pipe_lmh*: A value which describes which third of the screen teh gap in the pipes currently occupies.
 - Creating my state representation has been a gradual process. When I first began to implement my Q-Learning algorithm, I used only *delta_x* and *delta_y*. However, I quickly discovered that the bird's movement was highly dependent upon whether it was falling or not, so I added *is_falling* as well. Next, I was faced with the problem that the bird hit the ceiling or bottom too frequently. I found out that this was due to the bird not having enough information in its state in order to know its absolute position on the screen (delta_x and delta_y only give position relative to the pipe). In order to address this issue, I first used the bird's y position, but quickly found that this increased the size of my state space by too much. Instead, I added *bird_lmh* (credit to Justin Rushin) to tell the bird which third of the screen it was located on; this coarse granularity provided adequate enough information for the bird to avoid the top and bottom, but didn't increase the size of the state space too dramatically . Lastly, I added *pipe_lmh* to in order to tell the bird which third of the screen the next pipe opening was located on. 
    
Rewards
-------
 - Survival: 1.0 + (15.0 / (distance_to_pipe + .1))
 - Bird Passes Pipe: 10000.0
 - Death: -(1000.0 + (.5 * distance_to_pipe))
 - Hitting Floor/Ceiling: -5000

Gaussian Blur
-------------
In order to increase the training processes dramatically, I found it helpful to perform a gaussian blurring on the Q matrix every 50 training iterations. The blur on the matrix helped states 'infer' their values from states that were similar to their own. In this way, the birds could gain information regarding states that they had never seen before. Gaussian blurring represented the single largest performance increase in terms of learning speed.

Multiple Birds & Speed Up
-------------------------
In order to increase the speed of the learning process, I found it helpful to instantiate a number of birds that could learn concurrently using the same 'brain' (Q and V matrices).

In order to further increase the learning speed, I added a OVERALL_SPEED that could modify the game to play at increased speeds (credit to Peter Mash for figuring this out). Note: due to the use of the delta_frames in the some of the physics, it is important to note that actors trained at higher speeds will not perform well at lower ones. If you wish to run a bird trained on a faster brain on a lower speed, you must divide the FPS by the OVERALL_SPEED (example: if a bird is trainined on 10 overall speed, then to run it with overall speed 1, the FPS must be set to 6).

Intercede & Bias
----------------
In order to speed up the learning process of the birds, I added two optional parameters to my learning algorithm: 

First, *intercede* acts as a sort of safety net and will ensure that act() will never choose an option that will lead the bird to die from hitting the ceiling or the floor.
  
Second, bias helps break ties when the Q and V values for a given state are tied. When this is the case, the bird will find which third of the screen it is located on \bird_lmh\ and which thrid of the screen the pipe opening is located on \pipe_lmh\. If it is the case that the bird is on a lower third of the screen than is the pipe, the bird will opt to jump. Otherwise, the bird will opt to fall.

Of course, the use of these two methodologies represent a divergence from pure Q-Learning. However in reality, after training reaches a few iterations, neither method is used very often as there are almost never any ties, and the birds quickly learn to stay away from the floor and ceiling. These parameters merely act as a helpful boost in the beginning of the training process that allows the birds to learn extremely quickly.

Run Instructions
----------------
```
pip install -r requirements.txt  # install dependencies
python flappybird.py --actor_file 'amazing.pickle' --speed 10 --number_of_birds 1  # run at 10 times normal speed with 1 bird
```

There are a number of other parameters that you can set to customize your run:

--actor_file: file path to the actor file
--save_file: file path which you would like to write to (default is actor.pickle)
--speed: game speed (default 1)
--skip_frames: only allow the bird to make decisions on multiples of this frame
--number_of_birds: number of birds to play flappy bird with
--random_start_point: use random starting point for bird(s)
--alpha: learning rate (0.0 < alpha < 1.0)
--blur_freq: the number of runs between each gaussian blur
--sigma: sigma for guassian blur
--bias: introduce bias in the event of ties to help the bird learn faster
--intercede: do not allow the bird to suicide on floor/ceiling


Results
-------
The birds learn in exceptionally few iterations, and improve drastically over time. Performance boosts are especially prominent right after gaussian blurring (applied every 50 iterations by default). The run shown directly below is with 1000 birds running at 10 times the original speed. The final average pipes passed per bird was at 41.15, with the maximum consecutive pipes as 1139. 

```
python flappybird --speed 10 --skip_frame 1 --number_of_birds 1000 --rand True --alpha .8 --blur_freq 50 --sigma 2.0
              1: AVERAGE_PIPES:            0.00 	 RUN_PIPES:     0 	 OVERALL:     0
              2: AVERAGE_PIPES:            0.11 	 RUN_PIPES:    10 	 OVERALL:    10
              3: AVERAGE_PIPES:            0.30 	 RUN_PIPES:     4 	 OVERALL:    10
              4: AVERAGE_PIPES:            0.38 	 RUN_PIPES:     5 	 OVERALL:    10
              5: AVERAGE_PIPES:            0.38 	 RUN_PIPES:     3 	 OVERALL:    10
              6: AVERAGE_PIPES:            0.33 	 RUN_PIPES:     6 	 OVERALL:    10
              7: AVERAGE_PIPES:            0.34 	 RUN_PIPES:     3 	 OVERALL:    10
              8: AVERAGE_PIPES:            0.63 	 RUN_PIPES:     7 	 OVERALL:    10
              9: AVERAGE_PIPES:            0.63 	 RUN_PIPES:     8 	 OVERALL:    10
             10: AVERAGE_PIPES:            0.68 	 RUN_PIPES:     2 	 OVERALL:    10
             11: AVERAGE_PIPES:            0.77 	 RUN_PIPES:     6 	 OVERALL:    10
             12: AVERAGE_PIPES:            0.92 	 RUN_PIPES:     8 	 OVERALL:    10
             13: AVERAGE_PIPES:            0.86 	 RUN_PIPES:     5 	 OVERALL:    10
             14: AVERAGE_PIPES:            1.03 	 RUN_PIPES:    13 	 OVERALL:    13
             15: AVERAGE_PIPES:            1.01 	 RUN_PIPES:     3 	 OVERALL:    13
             16: AVERAGE_PIPES:            1.02 	 RUN_PIPES:     2 	 OVERALL:    13
             17: AVERAGE_PIPES:            1.13 	 RUN_PIPES:    15 	 OVERALL:    15
             18: AVERAGE_PIPES:            1.08 	 RUN_PIPES:     7 	 OVERALL:    15
             19: AVERAGE_PIPES:            1.08 	 RUN_PIPES:    21 	 OVERALL:    20
             20: AVERAGE_PIPES:            1.15 	 RUN_PIPES:     7 	 OVERALL:    20
             21: AVERAGE_PIPES:            1.26 	 RUN_PIPES:     7 	 OVERALL:    20
             22: AVERAGE_PIPES:            1.21 	 RUN_PIPES:     8 	 OVERALL:    20
             23: AVERAGE_PIPES:            1.27 	 RUN_PIPES:     4 	 OVERALL:    20
             24: AVERAGE_PIPES:            1.25 	 RUN_PIPES:     3 	 OVERALL:    20
             25: AVERAGE_PIPES:            1.35 	 RUN_PIPES:    10 	 OVERALL:    20
             26: AVERAGE_PIPES:            1.42 	 RUN_PIPES:    10 	 OVERALL:    20
             27: AVERAGE_PIPES:            1.44 	 RUN_PIPES:     5 	 OVERALL:    20
             28: AVERAGE_PIPES:            1.47 	 RUN_PIPES:    21 	 OVERALL:    21
             29: AVERAGE_PIPES:            1.56 	 RUN_PIPES:    14 	 OVERALL:    21
             30: AVERAGE_PIPES:            1.61 	 RUN_PIPES:     8 	 OVERALL:    21
             31: AVERAGE_PIPES:            1.66 	 RUN_PIPES:    10 	 OVERALL:    21
             32: AVERAGE_PIPES:            1.68 	 RUN_PIPES:     3 	 OVERALL:    21
             33: AVERAGE_PIPES:            1.63 	 RUN_PIPES:     9 	 OVERALL:    21
             34: AVERAGE_PIPES:            1.63 	 RUN_PIPES:     5 	 OVERALL:    21
             35: AVERAGE_PIPES:            1.63 	 RUN_PIPES:     7 	 OVERALL:    21
             36: AVERAGE_PIPES:            1.60 	 RUN_PIPES:    11 	 OVERALL:    21
             37: AVERAGE_PIPES:            1.73 	 RUN_PIPES:    16 	 OVERALL:    21
             38: AVERAGE_PIPES:            1.74 	 RUN_PIPES:    10 	 OVERALL:    21
             39: AVERAGE_PIPES:            1.71 	 RUN_PIPES:    14 	 OVERALL:    21
             40: AVERAGE_PIPES:            1.68 	 RUN_PIPES:     8 	 OVERALL:    21
             41: AVERAGE_PIPES:            1.74 	 RUN_PIPES:     8 	 OVERALL:    21
             42: AVERAGE_PIPES:            1.75 	 RUN_PIPES:     3 	 OVERALL:    21
             43: AVERAGE_PIPES:            1.75 	 RUN_PIPES:    23 	 OVERALL:    23
             44: AVERAGE_PIPES:            1.76 	 RUN_PIPES:     5 	 OVERALL:    23
             45: AVERAGE_PIPES:            1.73 	 RUN_PIPES:     7 	 OVERALL:    23
             46: AVERAGE_PIPES:            1.76 	 RUN_PIPES:    20 	 OVERALL:    23
             47: AVERAGE_PIPES:            1.84 	 RUN_PIPES:    16 	 OVERALL:    23
             48: AVERAGE_PIPES:            1.80 	 RUN_PIPES:     2 	 OVERALL:    23
             49: AVERAGE_PIPES:            1.80 	 RUN_PIPES:    12 	 OVERALL:    23
             50: AVERAGE_PIPES:            1.82 	 RUN_PIPES:    20 	 OVERALL:    23
             51: AVERAGE_PIPES:            1.88 	 RUN_PIPES:    43 	 OVERALL:    43
             52: AVERAGE_PIPES:            1.94 	 RUN_PIPES:     9 	 OVERALL:    43
             53: AVERAGE_PIPES:            1.98 	 RUN_PIPES:    29 	 OVERALL:    43
             54: AVERAGE_PIPES:            2.06 	 RUN_PIPES:    16 	 OVERALL:    43
             55: AVERAGE_PIPES:            2.13 	 RUN_PIPES:    31 	 OVERALL:    43
             56: AVERAGE_PIPES:            2.23 	 RUN_PIPES:    32 	 OVERALL:    43
             57: AVERAGE_PIPES:            2.33 	 RUN_PIPES:    89 	 OVERALL:    88
             58: AVERAGE_PIPES:            2.38 	 RUN_PIPES:    25 	 OVERALL:    88
             59: AVERAGE_PIPES:            2.51 	 RUN_PIPES:    49 	 OVERALL:    88
             60: AVERAGE_PIPES:            2.63 	 RUN_PIPES:    35 	 OVERALL:    88
             61: AVERAGE_PIPES:            2.71 	 RUN_PIPES:    20 	 OVERALL:    88
             62: AVERAGE_PIPES:            2.75 	 RUN_PIPES:    34 	 OVERALL:    88
             63: AVERAGE_PIPES:            2.88 	 RUN_PIPES:    60 	 OVERALL:    88
             64: AVERAGE_PIPES:            2.99 	 RUN_PIPES:    40 	 OVERALL:    88
             65: AVERAGE_PIPES:            3.05 	 RUN_PIPES:    46 	 OVERALL:    88
             66: AVERAGE_PIPES:            3.11 	 RUN_PIPES:    22 	 OVERALL:    88
             67: AVERAGE_PIPES:            3.16 	 RUN_PIPES:    34 	 OVERALL:    88
             68: AVERAGE_PIPES:            3.25 	 RUN_PIPES:    52 	 OVERALL:    88
             69: AVERAGE_PIPES:            3.30 	 RUN_PIPES:    21 	 OVERALL:    88
             70: AVERAGE_PIPES:            3.30 	 RUN_PIPES:    13 	 OVERALL:    88
             71: AVERAGE_PIPES:            3.38 	 RUN_PIPES:    31 	 OVERALL:    88
             72: AVERAGE_PIPES:            3.37 	 RUN_PIPES:    14 	 OVERALL:    88
             73: AVERAGE_PIPES:            3.38 	 RUN_PIPES:    15 	 OVERALL:    88
             74: AVERAGE_PIPES:            3.49 	 RUN_PIPES:    59 	 OVERALL:    88
             75: AVERAGE_PIPES:            3.54 	 RUN_PIPES:    34 	 OVERALL:    88
             76: AVERAGE_PIPES:            3.57 	 RUN_PIPES:    36 	 OVERALL:    88
             77: AVERAGE_PIPES:            3.65 	 RUN_PIPES:    75 	 OVERALL:    88
             78: AVERAGE_PIPES:            3.73 	 RUN_PIPES:    30 	 OVERALL:    88
             79: AVERAGE_PIPES:            3.82 	 RUN_PIPES:    75 	 OVERALL:    88
             80: AVERAGE_PIPES:            3.85 	 RUN_PIPES:    46 	 OVERALL:    88
             81: AVERAGE_PIPES:            3.87 	 RUN_PIPES:    45 	 OVERALL:    88
             82: AVERAGE_PIPES:            3.91 	 RUN_PIPES:    57 	 OVERALL:    88
             83: AVERAGE_PIPES:            3.96 	 RUN_PIPES:    33 	 OVERALL:    88
             84: AVERAGE_PIPES:            4.03 	 RUN_PIPES:    38 	 OVERALL:    88
             85: AVERAGE_PIPES:            4.14 	 RUN_PIPES:    62 	 OVERALL:    88
             86: AVERAGE_PIPES:            4.21 	 RUN_PIPES:    48 	 OVERALL:    88
             87: AVERAGE_PIPES:            4.24 	 RUN_PIPES:    39 	 OVERALL:    88
             88: AVERAGE_PIPES:            4.30 	 RUN_PIPES:    75 	 OVERALL:    88
             89: AVERAGE_PIPES:            4.31 	 RUN_PIPES:    31 	 OVERALL:    88
             90: AVERAGE_PIPES:            4.36 	 RUN_PIPES:    60 	 OVERALL:    88
             91: AVERAGE_PIPES:            4.35 	 RUN_PIPES:    19 	 OVERALL:    88
             92: AVERAGE_PIPES:            4.35 	 RUN_PIPES:    26 	 OVERALL:    88
             93: AVERAGE_PIPES:            4.38 	 RUN_PIPES:    26 	 OVERALL:    88
             94: AVERAGE_PIPES:            4.37 	 RUN_PIPES:    32 	 OVERALL:    88
             95: AVERAGE_PIPES:            4.41 	 RUN_PIPES:    19 	 OVERALL:    88
             96: AVERAGE_PIPES:            4.47 	 RUN_PIPES:    79 	 OVERALL:    88
             97: AVERAGE_PIPES:            4.51 	 RUN_PIPES:    65 	 OVERALL:    88
             98: AVERAGE_PIPES:            4.56 	 RUN_PIPES:    29 	 OVERALL:    88
             99: AVERAGE_PIPES:            4.57 	 RUN_PIPES:    25 	 OVERALL:    88
            100: AVERAGE_PIPES:            4.58 	 RUN_PIPES:     9 	 OVERALL:    88
            101: AVERAGE_PIPES:            4.70 	 RUN_PIPES:    68 	 OVERALL:    88
            102: AVERAGE_PIPES:            5.38 	 RUN_PIPES:   308 	 OVERALL:   307
            103: AVERAGE_PIPES:            6.38 	 RUN_PIPES:   656 	 OVERALL:   656
            104: AVERAGE_PIPES:            7.16 	 RUN_PIPES:   206 	 OVERALL:   656
            105: AVERAGE_PIPES:            7.85 	 RUN_PIPES:   311 	 OVERALL:   656
            106: AVERAGE_PIPES:            8.39 	 RUN_PIPES:   315 	 OVERALL:   656
            107: AVERAGE_PIPES:            8.87 	 RUN_PIPES:   218 	 OVERALL:   656
            108: AVERAGE_PIPES:            9.44 	 RUN_PIPES:   280 	 OVERALL:   656
            109: AVERAGE_PIPES:           10.15 	 RUN_PIPES:   348 	 OVERALL:   656
            110: AVERAGE_PIPES:           10.97 	 RUN_PIPES:   442 	 OVERALL:   656
            111: AVERAGE_PIPES:           12.00 	 RUN_PIPES:   364 	 OVERALL:   656
            112: AVERAGE_PIPES:           12.96 	 RUN_PIPES:   441 	 OVERALL:   656
            113: AVERAGE_PIPES:           13.79 	 RUN_PIPES:   313 	 OVERALL:   656
            114: AVERAGE_PIPES:           14.49 	 RUN_PIPES:   289 	 OVERALL:   656
            115: AVERAGE_PIPES:           15.23 	 RUN_PIPES:   224 	 OVERALL:   656
            116: AVERAGE_PIPES:           15.71 	 RUN_PIPES:   207 	 OVERALL:   656
            117: AVERAGE_PIPES:           16.35 	 RUN_PIPES:   279 	 OVERALL:   656
            118: AVERAGE_PIPES:           16.90 	 RUN_PIPES:   242 	 OVERALL:   656
            119: AVERAGE_PIPES:           17.22 	 RUN_PIPES:   314 	 OVERALL:   656
            120: AVERAGE_PIPES:           17.66 	 RUN_PIPES:   257 	 OVERALL:   656
            121: AVERAGE_PIPES:           18.46 	 RUN_PIPES:   326 	 OVERALL:   656
            122: AVERAGE_PIPES:           19.05 	 RUN_PIPES:   205 	 OVERALL:   656
            123: AVERAGE_PIPES:           19.70 	 RUN_PIPES:   344 	 OVERALL:   656
            124: AVERAGE_PIPES:           20.24 	 RUN_PIPES:   317 	 OVERALL:   656
            125: AVERAGE_PIPES:           20.80 	 RUN_PIPES:   359 	 OVERALL:   656
            126: AVERAGE_PIPES:           21.38 	 RUN_PIPES:   350 	 OVERALL:   656
            127: AVERAGE_PIPES:           21.92 	 RUN_PIPES:   308 	 OVERALL:   656
            128: AVERAGE_PIPES:           22.35 	 RUN_PIPES:   203 	 OVERALL:   656
            129: AVERAGE_PIPES:           22.94 	 RUN_PIPES:   338 	 OVERALL:   656
            130: AVERAGE_PIPES:           23.55 	 RUN_PIPES:   298 	 OVERALL:   656
            131: AVERAGE_PIPES:           24.18 	 RUN_PIPES:   365 	 OVERALL:   656
            132: AVERAGE_PIPES:           24.72 	 RUN_PIPES:   282 	 OVERALL:   656
            133: AVERAGE_PIPES:           25.30 	 RUN_PIPES:   293 	 OVERALL:   656
            134: AVERAGE_PIPES:           25.98 	 RUN_PIPES:   340 	 OVERALL:   656
            135: AVERAGE_PIPES:           26.53 	 RUN_PIPES:   298 	 OVERALL:   656
            136: AVERAGE_PIPES:           26.89 	 RUN_PIPES:   216 	 OVERALL:   656
            137: AVERAGE_PIPES:           27.42 	 RUN_PIPES:   247 	 OVERALL:   656
            138: AVERAGE_PIPES:           27.79 	 RUN_PIPES:   337 	 OVERALL:   656
            139: AVERAGE_PIPES:           28.42 	 RUN_PIPES:   588 	 OVERALL:   656
            140: AVERAGE_PIPES:           28.71 	 RUN_PIPES:   252 	 OVERALL:   656
            141: AVERAGE_PIPES:           29.31 	 RUN_PIPES:   317 	 OVERALL:   656
            142: AVERAGE_PIPES:           29.73 	 RUN_PIPES:   289 	 OVERALL:   656
            143: AVERAGE_PIPES:           30.12 	 RUN_PIPES:   316 	 OVERALL:   656
            144: AVERAGE_PIPES:           30.61 	 RUN_PIPES:   383 	 OVERALL:   656
            145: AVERAGE_PIPES:           31.17 	 RUN_PIPES:   313 	 OVERALL:   656
            146: AVERAGE_PIPES:           31.62 	 RUN_PIPES:   198 	 OVERALL:   656
            147: AVERAGE_PIPES:           32.02 	 RUN_PIPES:   236 	 OVERALL:   656
            148: AVERAGE_PIPES:           32.52 	 RUN_PIPES:   348 	 OVERALL:   656
            149: AVERAGE_PIPES:           32.80 	 RUN_PIPES:   343 	 OVERALL:   656
            150: AVERAGE_PIPES:           33.20 	 RUN_PIPES:   385 	 OVERALL:   656
            151: AVERAGE_PIPES:           34.08 	 RUN_PIPES:  1139 	 OVERALL:  1139
            152: AVERAGE_PIPES:           35.05 	 RUN_PIPES:   792 	 OVERALL:  1139
            153: AVERAGE_PIPES:           36.07 	 RUN_PIPES:   435 	 OVERALL:  1139
            154: AVERAGE_PIPES:           36.54 	 RUN_PIPES:   256 	 OVERALL:  1139
            155: AVERAGE_PIPES:           37.21 	 RUN_PIPES:   349 	 OVERALL:  1139
            156: AVERAGE_PIPES:           37.87 	 RUN_PIPES:   861 	 OVERALL:  1139
            157: AVERAGE_PIPES:           38.44 	 RUN_PIPES:   350 	 OVERALL:  1139
            158: AVERAGE_PIPES:           39.09 	 RUN_PIPES:   381 	 OVERALL:  1139
            159: AVERAGE_PIPES:           40.20 	 RUN_PIPES:   821 	 OVERALL:  1139
            160: AVERAGE_PIPES:           40.68 	 RUN_PIPES:   364 	 OVERALL:  1139
            161: AVERAGE_PIPES:           41.15 	 RUN_PIPES:   368 	 OVERALL:  1139
```

```
python flappybird --speed 10 --skip_frame 1 --number_of_birds 1000 --rand True --alpha .9 --blur_freq 10 --sigma 2.0
              1: AVERAGE_PIPES:            0.00 	 RUN_PIPES:     0 	 OVERALL:     0
              2: AVERAGE_PIPES:            0.09 	 RUN_PIPES:     2 	 OVERALL:     2
              3: AVERAGE_PIPES:            0.35 	 RUN_PIPES:     4 	 OVERALL:     4
              4: AVERAGE_PIPES:            0.34 	 RUN_PIPES:     7 	 OVERALL:     7
              5: AVERAGE_PIPES:            0.30 	 RUN_PIPES:     5 	 OVERALL:     7
              6: AVERAGE_PIPES:            0.36 	 RUN_PIPES:     6 	 OVERALL:     7
              7: AVERAGE_PIPES:            0.60 	 RUN_PIPES:    10 	 OVERALL:    10
              8: AVERAGE_PIPES:            0.60 	 RUN_PIPES:     4 	 OVERALL:    10
              9: AVERAGE_PIPES:            0.77 	 RUN_PIPES:     9 	 OVERALL:    10
             10: AVERAGE_PIPES:            0.82 	 RUN_PIPES:     3 	 OVERALL:    10
             11: AVERAGE_PIPES:            0.83 	 RUN_PIPES:     5 	 OVERALL:    10
             12: AVERAGE_PIPES:            0.77 	 RUN_PIPES:     4 	 OVERALL:    10
             13: AVERAGE_PIPES:            0.75 	 RUN_PIPES:     5 	 OVERALL:    10
             14: AVERAGE_PIPES:            0.73 	 RUN_PIPES:    14 	 OVERALL:    14
             15: AVERAGE_PIPES:            0.73 	 RUN_PIPES:    11 	 OVERALL:    14
             16: AVERAGE_PIPES:            0.94 	 RUN_PIPES:    14 	 OVERALL:    14
             17: AVERAGE_PIPES:            0.95 	 RUN_PIPES:    12 	 OVERALL:    14
             18: AVERAGE_PIPES:            1.18 	 RUN_PIPES:    33 	 OVERALL:    33
             19: AVERAGE_PIPES:            1.19 	 RUN_PIPES:    15 	 OVERALL:    33
             20: AVERAGE_PIPES:            1.23 	 RUN_PIPES:    19 	 OVERALL:    33
             21: AVERAGE_PIPES:            1.21 	 RUN_PIPES:     8 	 OVERALL:    33
             22: AVERAGE_PIPES:            1.32 	 RUN_PIPES:    15 	 OVERALL:    33
             23: AVERAGE_PIPES:            1.50 	 RUN_PIPES:    23 	 OVERALL:    33
             24: AVERAGE_PIPES:            1.68 	 RUN_PIPES:    33 	 OVERALL:    33
             25: AVERAGE_PIPES:            2.00 	 RUN_PIPES:    52 	 OVERALL:    51
             26: AVERAGE_PIPES:            2.24 	 RUN_PIPES:    22 	 OVERALL:    51
             27: AVERAGE_PIPES:            2.69 	 RUN_PIPES:    42 	 OVERALL:    51
             28: AVERAGE_PIPES:            2.72 	 RUN_PIPES:    20 	 OVERALL:    51
             29: AVERAGE_PIPES:            2.96 	 RUN_PIPES:    68 	 OVERALL:    68
             30: AVERAGE_PIPES:            3.47 	 RUN_PIPES:    69 	 OVERALL:    69
             31: AVERAGE_PIPES:            3.43 	 RUN_PIPES:    17 	 OVERALL:    69
             32: AVERAGE_PIPES:            3.45 	 RUN_PIPES:    55 	 OVERALL:    69
             33: AVERAGE_PIPES:            3.75 	 RUN_PIPES:    64 	 OVERALL:    69
             34: AVERAGE_PIPES:            4.28 	 RUN_PIPES:   137 	 OVERALL:   137
             35: AVERAGE_PIPES:            7.12 	 RUN_PIPES:   736 	 OVERALL:   736
             36: AVERAGE_PIPES:           16.48 	 RUN_PIPES:  1514 	 OVERALL:  1514
             37: AVERAGE_PIPES:           25.64 	 RUN_PIPES:  2270 	 OVERALL:  2270
             38: AVERAGE_PIPES:           35.12 	 RUN_PIPES:  1389 	 OVERALL:  2270
```

```
python flappybird --speed 10 --skip_frame 2 --number_of_birds 1000 --rand True --alpha .9 --blur_freq 10 --sigma 2.0 --bias False --intercede false
              1: AVERAGE_PIPES:            0.00 	 RUN_PIPES:     0 	 OVERALL:     0
              2: AVERAGE_PIPES:            0.04 	 RUN_PIPES:     1 	 OVERALL:     1
              3: AVERAGE_PIPES:            0.06 	 RUN_PIPES:     3 	 OVERALL:     3
              4: AVERAGE_PIPES:            0.07 	 RUN_PIPES:     4 	 OVERALL:     4
              5: AVERAGE_PIPES:            0.10 	 RUN_PIPES:     4 	 OVERALL:     4
              6: AVERAGE_PIPES:            0.20 	 RUN_PIPES:     7 	 OVERALL:     7
              7: AVERAGE_PIPES:            0.28 	 RUN_PIPES:     8 	 OVERALL:     8
              8: AVERAGE_PIPES:            0.29 	 RUN_PIPES:    11 	 OVERALL:    11
              9: AVERAGE_PIPES:            0.37 	 RUN_PIPES:     6 	 OVERALL:    11
             10: AVERAGE_PIPES:            0.41 	 RUN_PIPES:     7 	 OVERALL:    11
             11: AVERAGE_PIPES:            0.40 	 RUN_PIPES:     3 	 OVERALL:    11
             12: AVERAGE_PIPES:            0.45 	 RUN_PIPES:     8 	 OVERALL:    11
             13: AVERAGE_PIPES:            0.47 	 RUN_PIPES:     5 	 OVERALL:    11
             14: AVERAGE_PIPES:            0.54 	 RUN_PIPES:    16 	 OVERALL:    16
             15: AVERAGE_PIPES:            0.56 	 RUN_PIPES:    11 	 OVERALL:    16
             16: AVERAGE_PIPES:            0.69 	 RUN_PIPES:    17 	 OVERALL:    17
             17: AVERAGE_PIPES:            0.83 	 RUN_PIPES:    14 	 OVERALL:    17
             18: AVERAGE_PIPES:            0.93 	 RUN_PIPES:    15 	 OVERALL:    17
             19: AVERAGE_PIPES:            1.09 	 RUN_PIPES:    18 	 OVERALL:    18
             20: AVERAGE_PIPES:            1.10 	 RUN_PIPES:    14 	 OVERALL:    18
             21: AVERAGE_PIPES:            1.08 	 RUN_PIPES:     7 	 OVERALL:    18
             22: AVERAGE_PIPES:            1.07 	 RUN_PIPES:     6 	 OVERALL:    18
             23: AVERAGE_PIPES:            1.07 	 RUN_PIPES:     6 	 OVERALL:    18
             24: AVERAGE_PIPES:            1.04 	 RUN_PIPES:     6 	 OVERALL:    18
             25: AVERAGE_PIPES:            1.06 	 RUN_PIPES:     8 	 OVERALL:    18
             26: AVERAGE_PIPES:            1.14 	 RUN_PIPES:    12 	 OVERALL:    18
             27: AVERAGE_PIPES:            1.16 	 RUN_PIPES:    16 	 OVERALL:    18
             28: AVERAGE_PIPES:            1.24 	 RUN_PIPES:    27 	 OVERALL:    27
             29: AVERAGE_PIPES:            1.31 	 RUN_PIPES:    15 	 OVERALL:    27
             30: AVERAGE_PIPES:            1.40 	 RUN_PIPES:    21 	 OVERALL:    27
             31: AVERAGE_PIPES:            1.39 	 RUN_PIPES:     7 	 OVERALL:    27
             32: AVERAGE_PIPES:            1.36 	 RUN_PIPES:     5 	 OVERALL:    27
             33: AVERAGE_PIPES:            1.34 	 RUN_PIPES:     6 	 OVERALL:    27
             34: AVERAGE_PIPES:            1.35 	 RUN_PIPES:     8 	 OVERALL:    27
             35: AVERAGE_PIPES:            1.38 	 RUN_PIPES:    11 	 OVERALL:    27
             36: AVERAGE_PIPES:            1.41 	 RUN_PIPES:    10 	 OVERALL:    27
             37: AVERAGE_PIPES:            1.41 	 RUN_PIPES:    18 	 OVERALL:    27
             38: AVERAGE_PIPES:            1.51 	 RUN_PIPES:    22 	 OVERALL:    27
             39: AVERAGE_PIPES:            1.48 	 RUN_PIPES:    16 	 OVERALL:    27
             40: AVERAGE_PIPES:            1.49 	 RUN_PIPES:    16 	 OVERALL:    27
             41: AVERAGE_PIPES:            1.48 	 RUN_PIPES:     8 	 OVERALL:    27
             42: AVERAGE_PIPES:            1.47 	 RUN_PIPES:     9 	 OVERALL:    27
             43: AVERAGE_PIPES:            1.45 	 RUN_PIPES:     8 	 OVERALL:    27
             44: AVERAGE_PIPES:            1.46 	 RUN_PIPES:    14 	 OVERALL:    27
             45: AVERAGE_PIPES:            1.47 	 RUN_PIPES:    13 	 OVERALL:    27
             46: AVERAGE_PIPES:            1.49 	 RUN_PIPES:    18 	 OVERALL:    27
             47: AVERAGE_PIPES:            1.55 	 RUN_PIPES:    17 	 OVERALL:    27
             48: AVERAGE_PIPES:            1.62 	 RUN_PIPES:    20 	 OVERALL:    27
             49: AVERAGE_PIPES:            1.67 	 RUN_PIPES:    19 	 OVERALL:    27
             50: AVERAGE_PIPES:            1.70 	 RUN_PIPES:    32 	 OVERALL:    32
             51: AVERAGE_PIPES:            1.69 	 RUN_PIPES:     9 	 OVERALL:    32
             52: AVERAGE_PIPES:            1.67 	 RUN_PIPES:     6 	 OVERALL:    32
             53: AVERAGE_PIPES:            1.66 	 RUN_PIPES:     5 	 OVERALL:    32
             54: AVERAGE_PIPES:            1.66 	 RUN_PIPES:    10 	 OVERALL:    32
             55: AVERAGE_PIPES:            1.65 	 RUN_PIPES:    17 	 OVERALL:    32
             56: AVERAGE_PIPES:            1.68 	 RUN_PIPES:    23 	 OVERALL:    32
             57: AVERAGE_PIPES:            1.71 	 RUN_PIPES:    18 	 OVERALL:    32
             58: AVERAGE_PIPES:            1.75 	 RUN_PIPES:    20 	 OVERALL:    32
             59: AVERAGE_PIPES:            1.81 	 RUN_PIPES:    23 	 OVERALL:    32
             60: AVERAGE_PIPES:            1.90 	 RUN_PIPES:    48 	 OVERALL:    48
             61: AVERAGE_PIPES:            1.89 	 RUN_PIPES:     7 	 OVERALL:    48
             62: AVERAGE_PIPES:            1.88 	 RUN_PIPES:     8 	 OVERALL:    48
             63: AVERAGE_PIPES:            1.86 	 RUN_PIPES:     7 	 OVERALL:    48
             64: AVERAGE_PIPES:            1.86 	 RUN_PIPES:    10 	 OVERALL:    48
             65: AVERAGE_PIPES:            1.88 	 RUN_PIPES:    19 	 OVERALL:    48
             66: AVERAGE_PIPES:            1.87 	 RUN_PIPES:    15 	 OVERALL:    48
             67: AVERAGE_PIPES:            1.88 	 RUN_PIPES:    13 	 OVERALL:    48
             68: AVERAGE_PIPES:            1.89 	 RUN_PIPES:    18 	 OVERALL:    48
             69: AVERAGE_PIPES:            1.92 	 RUN_PIPES:    33 	 OVERALL:    48
             70: AVERAGE_PIPES:            1.95 	 RUN_PIPES:    29 	 OVERALL:    48
             71: AVERAGE_PIPES:            1.94 	 RUN_PIPES:     8 	 OVERALL:    48
             72: AVERAGE_PIPES:            1.93 	 RUN_PIPES:     5 	 OVERALL:    48
             73: AVERAGE_PIPES:            1.91 	 RUN_PIPES:     5 	 OVERALL:    48
             74: AVERAGE_PIPES:            1.89 	 RUN_PIPES:    12 	 OVERALL:    48
             75: AVERAGE_PIPES:            1.90 	 RUN_PIPES:    13 	 OVERALL:    48
             76: AVERAGE_PIPES:            1.92 	 RUN_PIPES:    18 	 OVERALL:    48
             77: AVERAGE_PIPES:            1.94 	 RUN_PIPES:    19 	 OVERALL:    48
             78: AVERAGE_PIPES:            1.97 	 RUN_PIPES:    32 	 OVERALL:    48
             79: AVERAGE_PIPES:            1.99 	 RUN_PIPES:    16 	 OVERALL:    48
             80: AVERAGE_PIPES:            2.02 	 RUN_PIPES:    27 	 OVERALL:    48
             81: AVERAGE_PIPES:            2.01 	 RUN_PIPES:     3 	 OVERALL:    48
             82: AVERAGE_PIPES:            1.99 	 RUN_PIPES:     4 	 OVERALL:    48
             83: AVERAGE_PIPES:            1.98 	 RUN_PIPES:     6 	 OVERALL:    48
             84: AVERAGE_PIPES:            1.98 	 RUN_PIPES:    25 	 OVERALL:    48
             85: AVERAGE_PIPES:            1.98 	 RUN_PIPES:    15 	 OVERALL:    48
             86: AVERAGE_PIPES:            1.97 	 RUN_PIPES:     6 	 OVERALL:    48
             87: AVERAGE_PIPES:            1.96 	 RUN_PIPES:    10 	 OVERALL:    48
             88: AVERAGE_PIPES:            1.97 	 RUN_PIPES:    16 	 OVERALL:    48
             89: AVERAGE_PIPES:            1.97 	 RUN_PIPES:    15 	 OVERALL:    48
             90: AVERAGE_PIPES:            1.98 	 RUN_PIPES:    15 	 OVERALL:    48
             91: AVERAGE_PIPES:            1.97 	 RUN_PIPES:     7 	 OVERALL:    48
             92: AVERAGE_PIPES:            1.96 	 RUN_PIPES:     5 	 OVERALL:    48
             93: AVERAGE_PIPES:            1.95 	 RUN_PIPES:     9 	 OVERALL:    48
             94: AVERAGE_PIPES:            1.95 	 RUN_PIPES:    10 	 OVERALL:    48
             95: AVERAGE_PIPES:            1.97 	 RUN_PIPES:    15 	 OVERALL:    48
             96: AVERAGE_PIPES:            1.98 	 RUN_PIPES:     9 	 OVERALL:    48
             97: AVERAGE_PIPES:            1.97 	 RUN_PIPES:    14 	 OVERALL:    48
             98: AVERAGE_PIPES:            1.98 	 RUN_PIPES:    28 	 OVERALL:    48
             99: AVERAGE_PIPES:            2.02 	 RUN_PIPES:    31 	 OVERALL:    48
            100: AVERAGE_PIPES:            2.07 	 RUN_PIPES:    23 	 OVERALL:    48
            101: AVERAGE_PIPES:            2.06 	 RUN_PIPES:    10 	 OVERALL:    48
            102: AVERAGE_PIPES:            2.06 	 RUN_PIPES:    12 	 OVERALL:    48
            103: AVERAGE_PIPES:            2.06 	 RUN_PIPES:    11 	 OVERALL:    48
            104: AVERAGE_PIPES:            2.06 	 RUN_PIPES:    15 	 OVERALL:    48
            105: AVERAGE_PIPES:            2.07 	 RUN_PIPES:    25 	 OVERALL:    48
            106: AVERAGE_PIPES:            2.07 	 RUN_PIPES:    15 	 OVERALL:    48
            107: AVERAGE_PIPES:            2.09 	 RUN_PIPES:    18 	 OVERALL:    48
            108: AVERAGE_PIPES:            2.09 	 RUN_PIPES:    13 	 OVERALL:    48
            109: AVERAGE_PIPES:            2.13 	 RUN_PIPES:    27 	 OVERALL:    48
            110: AVERAGE_PIPES:            2.15 	 RUN_PIPES:    18 	 OVERALL:    48
            111: AVERAGE_PIPES:            2.14 	 RUN_PIPES:     8 	 OVERALL:    48
            112: AVERAGE_PIPES:            2.12 	 RUN_PIPES:     6 	 OVERALL:    48
            113: AVERAGE_PIPES:            2.12 	 RUN_PIPES:     9 	 OVERALL:    48
            114: AVERAGE_PIPES:            2.11 	 RUN_PIPES:     6 	 OVERALL:    48
            115: AVERAGE_PIPES:            2.12 	 RUN_PIPES:    14 	 OVERALL:    48
            116: AVERAGE_PIPES:            2.14 	 RUN_PIPES:    22 	 OVERALL:    48
            117: AVERAGE_PIPES:            2.15 	 RUN_PIPES:    16 	 OVERALL:    48
            118: AVERAGE_PIPES:            2.15 	 RUN_PIPES:    12 	 OVERALL:    48
            119: AVERAGE_PIPES:            2.15 	 RUN_PIPES:    18 	 OVERALL:    48
            120: AVERAGE_PIPES:            2.17 	 RUN_PIPES:    27 	 OVERALL:    48
            121: AVERAGE_PIPES:            2.16 	 RUN_PIPES:     5 	 OVERALL:    48
            122: AVERAGE_PIPES:            2.15 	 RUN_PIPES:     7 	 OVERALL:    48
            123: AVERAGE_PIPES:            2.14 	 RUN_PIPES:     9 	 OVERALL:    48
            124: AVERAGE_PIPES:            2.14 	 RUN_PIPES:    10 	 OVERALL:    48
            125: AVERAGE_PIPES:            2.14 	 RUN_PIPES:     7 	 OVERALL:    48
            126: AVERAGE_PIPES:            2.13 	 RUN_PIPES:     7 	 OVERALL:    48
            127: AVERAGE_PIPES:            2.16 	 RUN_PIPES:    20 	 OVERALL:    48
            128: AVERAGE_PIPES:            2.17 	 RUN_PIPES:    16 	 OVERALL:    48
            129: AVERAGE_PIPES:            2.18 	 RUN_PIPES:    15 	 OVERALL:    48
            130: AVERAGE_PIPES:            2.20 	 RUN_PIPES:    20 	 OVERALL:    48
            131: AVERAGE_PIPES:            2.19 	 RUN_PIPES:    10 	 OVERALL:    48
            132: AVERAGE_PIPES:            2.18 	 RUN_PIPES:     7 	 OVERALL:    48
            133: AVERAGE_PIPES:            2.18 	 RUN_PIPES:     7 	 OVERALL:    48
            134: AVERAGE_PIPES:            2.19 	 RUN_PIPES:    12 	 OVERALL:    48
            135: AVERAGE_PIPES:            2.18 	 RUN_PIPES:    15 	 OVERALL:    48
            136: AVERAGE_PIPES:            2.19 	 RUN_PIPES:    21 	 OVERALL:    48
            137: AVERAGE_PIPES:            2.21 	 RUN_PIPES:    22 	 OVERALL:    48
            138: AVERAGE_PIPES:            2.22 	 RUN_PIPES:    16 	 OVERALL:    48
            139: AVERAGE_PIPES:            2.23 	 RUN_PIPES:    23 	 OVERALL:    48
            140: AVERAGE_PIPES:            2.24 	 RUN_PIPES:    18 	 OVERALL:    48
            141: AVERAGE_PIPES:            2.24 	 RUN_PIPES:     7 	 OVERALL:    48
            142: AVERAGE_PIPES:            2.23 	 RUN_PIPES:     4 	 OVERALL:    48
            143: AVERAGE_PIPES:            2.21 	 RUN_PIPES:     6 	 OVERALL:    48
            144: AVERAGE_PIPES:            2.21 	 RUN_PIPES:     9 	 OVERALL:    48
            145: AVERAGE_PIPES:            2.21 	 RUN_PIPES:    19 	 OVERALL:    48
            146: AVERAGE_PIPES:            2.21 	 RUN_PIPES:    14 	 OVERALL:    48
            147: AVERAGE_PIPES:            2.22 	 RUN_PIPES:    13 	 OVERALL:    48
            148: AVERAGE_PIPES:            2.23 	 RUN_PIPES:    14 	 OVERALL:    48
            149: AVERAGE_PIPES:            2.23 	 RUN_PIPES:    15 	 OVERALL:    48
            150: AVERAGE_PIPES:            2.25 	 RUN_PIPES:    20 	 OVERALL:    48
            151: AVERAGE_PIPES:            2.25 	 RUN_PIPES:     5 	 OVERALL:    48
            152: AVERAGE_PIPES:            2.24 	 RUN_PIPES:     8 	 OVERALL:    48
            153: AVERAGE_PIPES:            2.23 	 RUN_PIPES:     9 	 OVERALL:    48
            154: AVERAGE_PIPES:            2.23 	 RUN_PIPES:    14 	 OVERALL:    48
            155: AVERAGE_PIPES:            2.23 	 RUN_PIPES:    15 	 OVERALL:    48
            156: AVERAGE_PIPES:            2.24 	 RUN_PIPES:    26 	 OVERALL:    48
            157: AVERAGE_PIPES:            2.24 	 RUN_PIPES:    20 	 OVERALL:    48
            158: AVERAGE_PIPES:            2.25 	 RUN_PIPES:    10 	 OVERALL:    48
            159: AVERAGE_PIPES:            2.25 	 RUN_PIPES:     9 	 OVERALL:    48
            160: AVERAGE_PIPES:            2.26 	 RUN_PIPES:    20 	 OVERALL:    48
            161: AVERAGE_PIPES:            2.26 	 RUN_PIPES:    12 	 OVERALL:    48
            162: AVERAGE_PIPES:            2.25 	 RUN_PIPES:    14 	 OVERALL:    48
            163: AVERAGE_PIPES:            2.25 	 RUN_PIPES:    14 	 OVERALL:    48
            164: AVERAGE_PIPES:            2.24 	 RUN_PIPES:     9 	 OVERALL:    48
            165: AVERAGE_PIPES:            2.26 	 RUN_PIPES:    15 	 OVERALL:    48
            166: AVERAGE_PIPES:            2.28 	 RUN_PIPES:    19 	 OVERALL:    48
            167: AVERAGE_PIPES:            2.29 	 RUN_PIPES:    27 	 OVERALL:    48
            168: AVERAGE_PIPES:            2.30 	 RUN_PIPES:    17 	 OVERALL:    48
            169: AVERAGE_PIPES:            2.31 	 RUN_PIPES:    17 	 OVERALL:    48
            170: AVERAGE_PIPES:            2.32 	 RUN_PIPES:    19 	 OVERALL:    48
            171: AVERAGE_PIPES:            2.32 	 RUN_PIPES:    12 	 OVERALL:    48
            172: AVERAGE_PIPES:            2.31 	 RUN_PIPES:    18 	 OVERALL:    48
            173: AVERAGE_PIPES:            2.30 	 RUN_PIPES:     9 	 OVERALL:    48
            174: AVERAGE_PIPES:            2.31 	 RUN_PIPES:    14 	 OVERALL:    48
            175: AVERAGE_PIPES:            2.30 	 RUN_PIPES:     8 	 OVERALL:    48
            176: AVERAGE_PIPES:            2.31 	 RUN_PIPES:    16 	 OVERALL:    48
            177: AVERAGE_PIPES:            2.32 	 RUN_PIPES:    22 	 OVERALL:    48
            178: AVERAGE_PIPES:            2.33 	 RUN_PIPES:    17 	 OVERALL:    48
            179: AVERAGE_PIPES:            2.34 	 RUN_PIPES:    30 	 OVERALL:    48
            180: AVERAGE_PIPES:            2.37 	 RUN_PIPES:    41 	 OVERALL:    48
            181: AVERAGE_PIPES:            2.36 	 RUN_PIPES:     7 	 OVERALL:    48
            182: AVERAGE_PIPES:            2.35 	 RUN_PIPES:     6 	 OVERALL:    48
            183: AVERAGE_PIPES:            2.34 	 RUN_PIPES:     6 	 OVERALL:    48
            184: AVERAGE_PIPES:            2.34 	 RUN_PIPES:    32 	 OVERALL:    48
            185: AVERAGE_PIPES:            2.37 	 RUN_PIPES:    32 	 OVERALL:    48
            186: AVERAGE_PIPES:            2.38 	 RUN_PIPES:    22 	 OVERALL:    48
            187: AVERAGE_PIPES:            2.39 	 RUN_PIPES:    17 	 OVERALL:    48
            188: AVERAGE_PIPES:            2.40 	 RUN_PIPES:    17 	 OVERALL:    48
            189: AVERAGE_PIPES:            2.42 	 RUN_PIPES:    30 	 OVERALL:    48
            190: AVERAGE_PIPES:            2.43 	 RUN_PIPES:    28 	 OVERALL:    48
            191: AVERAGE_PIPES:            2.42 	 RUN_PIPES:    10 	 OVERALL:    48
            192: AVERAGE_PIPES:            2.42 	 RUN_PIPES:     6 	 OVERALL:    48
            193: AVERAGE_PIPES:            2.41 	 RUN_PIPES:    11 	 OVERALL:    48
            194: AVERAGE_PIPES:            2.42 	 RUN_PIPES:    12 	 OVERALL:    48
            195: AVERAGE_PIPES:            2.42 	 RUN_PIPES:    15 	 OVERALL:    48
            196: AVERAGE_PIPES:            2.42 	 RUN_PIPES:    15 	 OVERALL:    48
            197: AVERAGE_PIPES:            2.44 	 RUN_PIPES:    34 	 OVERALL:    48
            198: AVERAGE_PIPES:            2.45 	 RUN_PIPES:    20 	 OVERALL:    48
            199: AVERAGE_PIPES:            2.47 	 RUN_PIPES:    17 	 OVERALL:    48
            200: AVERAGE_PIPES:            2.48 	 RUN_PIPES:    38 	 OVERALL:    48
