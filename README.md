# RLProject
Simple benchmark for RL algorithms.

---

## Installation: </br> ##
python -m pip install -r requirements.txt

## Original Version: </br> ##
A logging farm game. </br>
Run <b>python main.py</b> to start playing. </br>
Control of game: </br>

|Keyboard|Action|
|:---:|:---:|
|Up Arrow|Move cursor up|
|Down Arrow|Move cursor Down|
|Left Arrow|Move cursor left|
|Right Arrow|Move cursor right|
|c|Cut tree|
|n|Next year|
|0|Cut all trees with age 0|
|1|Cut all trees with age 1|
|2|Cut all trees with age 2|
|3|Cut all trees with age 3|
|4|Cut all trees with age 4|
|5|Cut all trees with age 5|
|6|Cut all trees with age 6|
|7|Cut all trees with age 7|

Final goal of game: Obtain as much profit(money) as possible in the limited number of years.

## Version 1 </br> ##
The simplest version. Trees with age 7 will seed seeds around them. </br>
Environment is in Tree_env_1.py. </br>
Training and Evaluation: <b>python [algorithm].py</b>

## Version 1 Value_of_GHG</br> ##
Added a threshold of CO2 absorption that needs to surpass. </br>
Environment is in Tree_env_1.py. </br>
Training and Evaluation: <b>python [algorithm].py</b>

## Version 1.2 </br> ##
Combined value of tree and CO2 into one reward. </br>
Environment is in Tree_env_1.py. </br>
Training and Evaluation: <b>python [algorithm].py</b>

## Version 1.5 </br> ##
Took soil fertility into consideration. </br>
Environment is in Tree_env_1.py. </br>
Training and Evaluation: <b>python [algorithm].py</b>

## Version 2.0 </br> ##
Integrated all ideas of previous versions. </br>
Environment is in Tree_env_1.py. </br>
Training and Evaluation: <b>python [algorithm].py</b>

