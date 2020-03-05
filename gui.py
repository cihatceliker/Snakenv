from snake import POSSIBLE_DIRECTIONS_TO_GO, EMPTY, FOOD, EGG
from tkinter import Frame
import time
import threading
import pickle
import numpy as np
import random


BACKGROUND_COLOR = "#000"

COLORS = {
    EGG: "#C1895B",
    FOOD: "#EB252A",
    EMPTY: "#313B74"
}

def get_color(bg=COLORS[EMPTY]):
    r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
    noise = 80
    r = min(255, max(0, 255-r+random.randint(-noise, noise)))
    g = min(255, max(0, 255-g+random.randint(-noise, noise)))
    b = min(255, max(0, 255-b+random.randint(-noise, noise)))
    r = hex(r).split('x')[-1]
    g = hex(g).split('x')[-1]
    b = hex(b).split('x')[-1]
    return "#"+r+g+b


class GameGrid(Frame):

    def __init__(self, env, grid_padding=1, frame_padding=0, speed=0.05, size=800):
        Frame.__init__(self)
        self.env = env
        self.grid_padding = grid_padding
        self.frame_padding = frame_padding
        self.speed = speed
        self.size = size
        self.env.reset()
        self.row = env.row
        self.col = env.col
        self.grid()
        self.master.title('Snakenv')
        self.pause = False
        self.master.bind("<Key>", self.key_down)
        self.grid_cells = []
        self.init_grid()
        self.update_grid_cells()
        self.start()
        self.mainloop()

    def update(self):
        pickle_in = open("w.snk","rb"); agent = pickle.load(pickle_in)
        obs = self.env.reset()
        curr = time.time()
        score = 0
        while self.env.snakes or self.env.eggs:
            if curr + self.speed < time.time():
                action_list = []
                for state, reward, _, _ in obs:
                    action = agent.select_action(state)
                    action_list.append(action)
                    score += reward
                obs_ = self.env.step(action_list)
                for _,_,_,info in obs_:
                    if info != "": 
                        print(info)
                obs = obs_
                self.update_grid_cells()
                curr = time.time()
        print("final score", score)
        self.quit()

    def start(self):
        threading.Thread(target=self.update).start()

    def prepare_board(self):
        # applying frame padding
        if self.frame_padding == 0:
            self.board = self.env.board
            return
        self.board = np.zeros((self.row+2*self.frame_padding, self.col+2*self.frame_padding))
        self.board[self.frame_padding:-self.frame_padding,self.frame_padding:-self.frame_padding] = self.env.board
        self.board[-self.frame_padding:,self.frame_padding:-self.frame_padding] = self.env.board[:self.frame_padding,:]
        self.board[:self.frame_padding,self.frame_padding:-self.frame_padding] = self.env.board[-self.frame_padding:,:]
        self.board[self.frame_padding:-self.frame_padding,-self.frame_padding:] = self.env.board[:,:self.frame_padding]
        self.board[self.frame_padding:-self.frame_padding,:self.frame_padding] = self.env.board[:,-self.frame_padding:]
        self.board[:self.frame_padding,:self.frame_padding] = self.env.board[-self.frame_padding:, -self.frame_padding:]
        self.board[:self.frame_padding,-self.frame_padding:] = self.env.board[-self.frame_padding:, :self.frame_padding]
        self.board[-self.frame_padding:,:self.frame_padding] = self.env.board[:self.frame_padding, -self.frame_padding:]
        self.board[-self.frame_padding:,-self.frame_padding:] = self.env.board[:self.frame_padding, :self.frame_padding]

    def init_grid(self):
        assert self.row == self.col # for now
        self.prepare_board()
        background = Frame(self, bg=BACKGROUND_COLOR, width=self.size, height=self.size)
        background.grid()
        for i in range(len(self.board)):
            grid_row = []
            for j in range(len(self.board)):
                cell = Frame(background, bg=BACKGROUND_COLOR, \
                    width=self.size/(self.col+self.frame_padding), height=self.size/(self.row+self.frame_padding))
                cell.grid(row=i, column=j, padx=self.grid_padding, pady=self.grid_padding)
                grid_row.append(cell)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        self.prepare_board()
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                x, y = i-self.frame_padding, j-self.frame_padding
                curr = int(self.board[i, j])
                if curr == EMPTY or curr == FOOD or curr == EGG:
                    self.grid_cells[i][j].configure(bg=COLORS[curr])
                else:
                    for snake in self.env.snakes:
                        if curr != snake.id: continue
                        if (x, y) == snake.head:
                            self.grid_cells[i][j].configure(bg="#000")
                        else:
                            # comment seed to see the rainbow mode
                            random.seed(snake.id)
                            self.grid_cells[i][j].configure(bg=get_color())
        self.update_idletasks()
    
    def key_down(self, event):
        self.snake = self.env.snakes[0]
        self.commands = {
            "'a'": POSSIBLE_DIRECTIONS_TO_GO[self.snake.direction][2],
            "'d'": POSSIBLE_DIRECTIONS_TO_GO[self.snake.direction][1]
        }
        key = repr(event.char)

        if key == "'q'":
            self.quit()
        if key == "'s'":
            self.speed += 0.005
        if key == "'w'":
            self.speed -= 0.005
        if key == "' '":
            if not self.pause:
                self.prev = self.speed
                self.speed = 100000
            else:
                self.speed = self.prev
            self.pause = not self.pause
        #if key in self.commands: self.snake.direction = self.commands[repr(event.char)]