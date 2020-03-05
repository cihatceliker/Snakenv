import numpy as np
from snake import Snake, EMPTY, FOOD, POSSIBLE_DIRECTIONS_TO_GO
import sys
from gui import GameGrid


class Environment:

    def __init__(self, row, col, num_snakes, throw_food_every):
        self.row = row
        self.col = col
        self.num_snakes = num_snakes
        self.throw_food_every = throw_food_every
    
    def reset(self):
        self.board = np.ones((self.row, self.col), dtype=np.int8) * EMPTY
        self.snakes = []
        self.eggs = []
        self.to_be_killed = []
        self.snake_ids = 1
        self.tick = 0
        for i in range(self.num_snakes):
            snake = Snake((self.row // 2, self.col // (self.num_snakes + 1) * self.snake_ids), self.snake_ids, self)
            self.add_snake(snake)
        self.throw_food()
        experience_list = []
        for snake in self.snakes:
            experience_list.append((snake.give_state(), 0., False, ""))
        return experience_list

    def step(self, action_list):
        self.tick += 1
        if self.tick % self.throw_food_every == 0:
            self.throw_food()
        for egg in self.eggs:
            egg.step()
        for i in range(min(len(action_list),len(self.snakes))):
            self.snakes[i].direction = POSSIBLE_DIRECTIONS_TO_GO[self.snakes[i].direction][action_list[i]]
        experience_list = []
        for snake in self.snakes:
            snake.pre_step()
        for snake in self.snakes:
            experience_list.append(snake.step())
        for snake in self.to_be_killed:
            if snake in self.snakes:
                self.snakes.remove(snake)
        self.to_be_killed = []
        return experience_list

    def throw_food(self):
        x_ = np.random.randint(self.row)
        y_ = np.random.randint(self.col)
        while self.board[x_, y_] != EMPTY:
            x_ = np.random.randint(self.row)
            y_ = np.random.randint(self.col)
        self.board[x_, y_] = FOOD

    def add_snake(self, snake):
        self.snakes.append(snake)
        self.snake_ids += 1

    def render(self):
        GameGrid(self)
    

if __name__ == "__main__":
    env = Environment(row=24, col=24, num_snakes=6, throw_food_every=20)
    env.render()