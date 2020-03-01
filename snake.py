import numpy as np
import sys

EMPTY = -1
FOOD = -2
EGG = -3
THROW_FOOD_EVERY = 20

# 0 -> up
# 1 -> right
# 2 -> down
# 3 -> left
dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# relative to itself
# [go straight, turn right, turn left]
POSSIBLE_DIRECTIONS_TO_GO = {
    0: [0, 1, 3],
    1: [1, 2, 0],
    2: [2, 3, 1],
    3: [3, 0, 2]
}
#[back, back left, left, front left, front, front right, right, back right]
DIRECTIONS_TO_LOOK = {
    0: [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)],
    1: [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)],
    2: [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)],
    3: [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
}
"""
# relative to gamearea
POSSIBLE_DIRECTIONS_TO_GO = {
    0: [0, 1, 2, 3],
    1: [0, 1, 2, 3],
    2: [0, 1, 2, 3],
    3: [0, 1, 2, 3]
}
DIRECTIONS_TO_LOOK = {
    0: [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)],
    1: [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)],
    2: [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)],
    3: [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
}
"""

class Snake():

    def __init__(self, start, id_, env, radius=8, egg_growth_limit=10, snake_growth_limit=10, pregnancy_time=4, hunger_threshold=100):
        self.env = env
        self.id = id_
        self.snake_growth_limit = snake_growth_limit
        self.egg_growth_limit = egg_growth_limit
        self.pregnancy_time = pregnancy_time
        self.hunger_threshold = hunger_threshold
        self.radius = radius
        self.head = start
        self.tail = start
        self.body = [self.head]
        self.env.board[start] = self.id
        self.direction = 0
        self.food_queue = []
        self.hunger = 0
        self.giving_birth = False
        
    def step(self):
        if len(self.body) == self.snake_growth_limit:
            self.giving_birth = True
            self.preg_count = 0
            self.birth_index = len(self.body) - self.pregnancy_time
            self.birth_location = self.body[self.birth_index]
            self.clean_food_queue(self.birth_index)
        if self.giving_birth:
            self.preg_count += 1
            if self.preg_count == self.pregnancy_time:
                egg = Egg(self, self.birth_location, self.env, self.egg_growth_limit)
                self.tail = self.body[-1]
                self.giving_birth = False
            else:
                self.env.board[self.body.pop()] = EMPTY
                self.tail = self.body[-1]
            return self.observe(), False, 0, ""
        return self.move()
    
    def move(self):
        cur_dir = dirs[self.direction][0] + self.head[0], \
                  dirs[self.direction][1] + self.head[1]
        cur_dir = self.check_wall_hit(cur_dir)
        self.hunger += 1
        reward = 0
        num = self.env.board[cur_dir]
        if num == self.id or \
            self.hunger > self.hunger_threshold:
            return self.kill()
        elif num == FOOD:
            self.hunger = 0 # -= 50 # len(self.body) ##### 
            self.food_queue.append(cur_dir)
            reward += 1
        elif num == EGG:
            self.hunger = 0
            self.food_queue.append(cur_dir)
        elif num != EMPTY:
            # it finds the snake its eating
            for snake in self.env.snakes:
                if num == snake.id:
                    if cur_dir == snake.head:
                        if len(self.body) >= len(snake.body):
                            snake.kill()
                            reward += 1
                        else:
                            return self.kill()
                    else:
                        snake.eat_body_from(cur_dir)
                    break
        self.body.insert(0, cur_dir)
        self.head = cur_dir
        self.env.board[cur_dir] = self.id
        if self.food_queue and self.tail == self.food_queue[0]:
            self.food_queue.pop(0)
        else:
            self.env.board[self.body.pop()] = EMPTY
            self.tail = self.body[-1]
        return self.observe(), reward, False, ""

    def kill(self):
        for body_part in self.body:
            self.env.board[body_part] = FOOD
        self.env.snakes.remove(self)
        return self.observe(), -1, True, ""

    def observe(self):
        brain_food = []
        for look_to in DIRECTIONS_TO_LOOK[self.direction]:
            signals = self.check_dir(*look_to)
            brain_food = [*brain_food, *signals]
        # last addition to parameters is hunger
        hunger_signal =  self.hunger / self.hunger_threshold
        brain_food.append(hunger_signal)
        return brain_food

    def check_dir(self, i, j):
        #params -> [food_dist, egg_dist, self_dist, other_dist, head_dist, size_comparison]
        params = [0] * 6
        is_found = [False] * 4
        head_found = False
        check_it = [FOOD, EGG, self.id, 999999] # last is just a placeholder
        x_, y_ = self.head
        x_ += i; y_ += j
        distance = 1
        while distance <= self.radius:
            x_, y_ = self.check_wall_hit((x_, y_))
            current_num = self.env.board[x_, y_]
            if current_num != EMPTY:
                for k in range(len(check_it)):
                    if not is_found[k]:
                        if check_it[k] == current_num or k == len(check_it) - 1:
                            params[k] = 1 / distance
                            is_found[k] = True
                            break
                if not head_found:
                    # it will check for any head on its way
                    for snake in self.env.snakes:
                        if snake.head == (x_, y_):
                            params[4] = 1 / distance
                            head_found = True
                            params[5] = -1 if len(self.body) < len(snake.body) else 1
                            break
            x_ += i
            y_ += j
            distance += 1
        return params

    def eat_body_from(self, start):
        ################
        if self.giving_birth:
            self.kill()
            return
        ################
        for i in range(len(self.body)):
            if self.body[i] == start:
                for j in range(i + 1, len(self.body)):
                    self.env.board[self.body[j]] = FOOD
                self.clean_food_queue(i)
                self.body = self.body[:i]
                self.tail = self.body[-1]
                return

    def clean_food_queue(self, start):
        # not optimal
        for j in range(start, len(self.body)):
            for k in range(len(self.food_queue)):
                if self.body[j] == self.food_queue[k]:
                    self.food_queue.pop(k)
                    break

    def check_wall_hit(self, cur_dir):
        if cur_dir[0] < 0: cur_dir = self.env.row - 1, cur_dir[1]
        elif cur_dir[0] == self.env.row: cur_dir = 0, cur_dir[1]
        if cur_dir[1] < 0: cur_dir = cur_dir[0], self.env.col - 1
        elif cur_dir[1] == self.env.col: cur_dir = cur_dir[0], 0
        return cur_dir


class Egg:

    def __init__(self, parent, loc, env, growth_limit):
        self.growth_limit = growth_limit
        self.growth = 0
        self.parent = parent
        self.loc = loc
        self.env = env
        self.env.board[loc] = EGG
        self.env.eggs.append(self)
    
    def step(self):
        self.growth += 1
        self.env.board[self.loc] = EGG
        if self.growth == self.growth_limit:
            self.break_the_egg()
        
    def break_the_egg(self):
        snake = Snake(self.loc, self.env.snake_ids, self.env)
        self.env.add_snake(snake)
        self.env.eggs.remove(self)