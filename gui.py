from tkinter import Frame, Label, CENTER
import time
import threading
from snake import POSSIBLE_DIRECTIONS_TO_GO, EMPTY, FOOD, EGG
import pickle
import random

SPEED = 0.06
SIZE = 600
GRID_PADDING = 5
BACKGROUND_COLOR = "#457a01"

COLORS = {
    EGG: "#fce2db",
    FOOD: "#c1094d",
    EMPTY: "#457a01"
}

KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"

class GameGrid(Frame):

    def __init__(self, env):
        Frame.__init__(self)
        self.env = env
        self.env.reset()
        self.speed = SPEED
        self.row = env.row
        self.col = env.col
        self.grid()
        self.master.title('SnakeWithBrain')
        self.master.bind("<Key>", self.key_down)
        self.r = lambda: random.randint(0,255)
        self.grid_cells = []
        self.init_grid()
        self.update_grid_cells()
        self.start()
        self.mainloop()

    def update(self):
        pickle_in = open("w.pickle","rb"); agent = pickle.load(pickle_in)
        obs = self.env.reset()
        score = 0
        curr = time.time()
        while self.env.snakes or self.env.eggs:
            if curr + SPEED < time.time():
                action_list = []
                for state, _, _, _ in obs:
                    action = agent.select_action(state)
                    action_list.append(action)
                obs_ = self.env.step(action_list)
                obs = obs_
                self.update_grid_cells()
                curr = time.time()
        self.quit()

    def start(self):
        threading.Thread(target=self.update).start()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR, width=SIZE, height=SIZE)
        background.grid()
        for i in range(self.row):
            grid_row = []
            for j in range(self.col):
                cell = Frame(background, bg=BACKGROUND_COLOR, width=SIZE/self.col, height=SIZE/self.row)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                grid_row.append(cell)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(self.row):
            for j in range(self.col):
                curr = int(self.env.board[i, j])
                
                if curr == EMPTY or curr == FOOD or curr == EGG:
                    self.grid_cells[i][j].configure(bg=COLORS[curr])
                else:
                    for snake in self.env.snakes:
                        if curr != snake.id: continue
                        if (i, j) == snake.head:
                            self.grid_cells[i][j].configure(bg="#000")
                        else:
                            random.seed(snake.id)
                            self.grid_cells[i][j].configure(bg='#%02X%02X%02X' % (self.r(), self.r(), self.r()))
        self.update_idletasks()
    
    def key_down(self, event):
        self.snake = self.env.snakes[0]
        self.commands = {
            KEY_LEFT: POSSIBLE_DIRECTIONS_TO_GO[self.snake.direction][2],
            KEY_RIGHT: POSSIBLE_DIRECTIONS_TO_GO[self.snake.direction][1]
        }
        key = repr(event.char)
        if key == "'q'":
            self.quit()
        if key in self.commands: self.snake.direction = self.commands[repr(event.char)]