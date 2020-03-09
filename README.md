# Snakenv

Snakenv is a customized version of the classic snake game and has an AI in it. Here are the rules:

 - There are multiple snakes.
 - Once they reach a certain length, they lay an egg and a couple of steps later egg will turn into a new snake. While they are laying an egg, they stop and shrink a bit and they are vulnerable.
 - Snakes can eat each other. If a snake tries to eat other's head, the smaller snake will die. If it tries to eat other's bodies, the other's body from that point to its tail will become food, but the other snake will survive with what's left of its body.
 - Snakes have hunger and hunger threshold. If their hunger goes past the hunger threshold, they die, so they have to eat.
 - The environment throws food to the game area at regular intervals.
 - Eating food zeroes the hunger. Eating other snakes or an egg, have a different effect on hunger.


# 

## AI

In this project, a little bit modified version of **[Deep-Q Learning](https://arxiv.org/abs/1312.5602)** is used.  Here's 
how it works briefly:

 1. Initialize the memory and the brain.
 2. While at least a snake lives:
	 
	 -  Observe states from all of the snakes'.
	 -  Choose an action for each of them. 
	 -  Play the actions, gather the next states and the rewards.
	 -  Store all of the experiences to the memory as a tuple of **<** *state, action, reward, next_state, done* **>**.
	 -  The brain learns from the randomly chosen experiences in that memory.
 3. Repeat the second step for every iteration.


## Training

#### Observation and Action Space

Snakes have sensors in 8 directions relative to their head position and moving direction. They also have a radius which restricts their vision. 
There are 7 parameters per direction. They look for:
- Food
- Egg
- Body remains
- Body part of itself
- Other snakes' body
- Other snakes' head
- If there is a head found, is it bigger than itself

If they found any of these, they set that parameter to the distance where they found it. This makes 56 inputs in total. Lastly, they have *hunger* as the last input.
All of these go into the brain as an observation after that, the brain gives an output. Possible actions are:
- Turn left
- Go straight
- Turn right


### Controls

 - X -> Game-speed ++
 - Z -> Game-speed --
 - Q -> quit
 - P -> pause/continue

    To visualize a snakes' brain, pause the game and choose a snake.
    To control a snake:
     - Press "c" after you choose a snake
     - Continue the game and control the snake with W, A, S, D

 Just run the *gui.py* file to watch them play.
 

#### Scores graph:
![alt text](/img/avg_scores.png)

# 

## Some screens

![alt text](/img/ss1.gif)
# 
![alt text](/img/ss2.gif)
# 

