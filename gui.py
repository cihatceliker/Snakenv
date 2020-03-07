from snake import CONTROLLER_MAPPING, POSSIBLE_DIRECTIONS_TO_GO, EMPTY, FOOD, EGG, REMAINS
from tkinter import Frame, Canvas, Tk
import time
import threading
import pickle
from environment import Environment
import numpy as np
import random
import sys

BACKGROUND_COLOR = "#000"
NEURON_COLOR = "red"

COLORS = {
    EGG: "#C1895B",
    FOOD: "#EB252A",
    REMAINS: "#A18201",
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


class GameGrid():

    def __init__(self, env, speed=0.05, size=720):
        self.root = Tk()
        self.root.configure(background=BACKGROUND_COLOR)
        self.left_frame = Frame(self.root, width=size, height=size, bg=BACKGROUND_COLOR)
        self.left_frame.grid(row=0, column=0)
        self.right_frame = Frame(self.root, width=size, height=size, bg=BACKGROUND_COLOR)
        self.right_frame.grid(row=0, column=1)

        self.game = Canvas(self.left_frame, width=size, height=size, bg=BACKGROUND_COLOR)
        self.game.pack()
        self.visual = Canvas(self.right_frame, width=size, height=size, bg=BACKGROUND_COLOR)
        self.visual.pack()

        pickle_in = open("w.snk","rb")
        self.agent = pickle.load(pickle_in)
        self.env = env
        self.env.reset()
        self.speed = speed
        self.size = size
        self.rectangle_size = size/self.env.row
        self.pause = False
        self.chosen_one = None
        self.controller = False

        self.init_board()
        self.update_board()
        self.draw_brain()

        self.root.title('Snakenv')
        self.commands = {"'w'": 0,"'a'": 3,"'s'": 2,"'d'": 1}
        self.root.bind("<Key>", self.key_down)
        self.root.bind('<Button-1>', self.on_click)
        self.start()
        self.root.mainloop()

    def start(self):
        threading.Thread(target=self.run_game).start()

    def run_game(self):
        obs = self.env.reset()
        while self.env.snakes or self.env.eggs:
            if not self.pause:
                action_list = []
                for state, reward, _, _ in obs:
                    action = self.agent.select_action(state)
                    action_list.append(action)
                obs_ = self.env.step(action_list, None if not self.chosen_one or not self.controller else self.chosen_one.id)
                for _,_,_,info in obs_:
                    if info != "":
                        print(info)
                obs = obs_
                if self.chosen_one not in self.env.snakes:
                    self.chosen_one = None
                    self.controller = False
                    self.visual.configure(background=BACKGROUND_COLOR)
                self.update_board()
                self.update_brain()
                time.sleep(max(self.speed, 0))

    def update_board(self):
        for i in range(self.env.row):
            for j in range(self.env.col):
                curr = int(self.env.board[i, j])
                if curr in COLORS:
                    color = COLORS[curr]
                else:
                    for snake in self.env.snakes:
                        if curr != snake.id: continue
                        if (i, j) == snake.head:
                            color = "black"
                        else:
                            # comment seed to see the rainbow mode
                            random.seed(snake.id)
                            color = get_color()
                self.game.itemconfig(self.game_area[i][j], fill=color)

    def init_board(self):
        def draw_rectangle(to, x1, y1, sz, color):
            return to.create_rectangle(x1, y1, x1+sz, y1+sz, fill=color, width=0)
        self.game_area = []
        for i in range(self.env.row):
            row = []
            for j in range(self.env.col):
                curr = int(self.env.board[i, j])
                if curr in COLORS:
                    color = COLORS[curr]
                else:
                    for snake in self.env.snakes:
                        if curr != snake.id: continue
                        if (i, j) == snake.head:
                            color = "black"
                        else:
                            # comment seed to see the rainbow mode
                            random.seed(snake.id)
                            color = get_color()
                
                rect = draw_rectangle(self.game, j*self.rectangle_size, i*self.rectangle_size, self.rectangle_size, color)
                row.append(rect)
            self.game_area.append(row)

    def draw_brain(self, radius=10, padding=1):    
        layers = []
        brain = self.agent.local_Q.state_dict().items()
        for k, v in brain:
            if not "bias" in k:
                layers.append(v.size(1))
        layers.append(self.agent.num_actions)
        num_layers = len(layers)
        offset_x = []
        for i in range(num_layers):
            num_neurons = layers[i]
            start = (self.size-num_neurons*radius-(num_neurons-1)*padding)/2
            offset_x.append(start)
        layer_size = self.size / num_layers
        x_coordinates = []
        for i in range(num_layers):
            x_coordinates.append((i+1)*layer_size-layer_size/2-radius)
        y_coordinates = []
        for layer in range(num_layers):
            for i in range(layers[layer]):
                coord = offset_x[layer]+(radius+padding)*i
                y_coordinates.append(coord)
        coords = {}
        for i in x_coordinates:
            coords[i] = []
        k = 0
        j = 0
        for i in layers:
            for _ in range(int(i)):
                coords[x_coordinates[j]].append(y_coordinates[k])
                k+=1
            j+=1
        c = []
        for k, v in coords.items():
            for i in v:
                c.append((k, i))
        coord_layers = []
        k = 0
        for i in layers:
            coord_layers.append([])
            for _ in range(i):
                coord_layers[-1].append(c[k])
                k+=1
        self.neurons = []
        self.lines = []
        for i in range(len(coord_layers)-1):
            for j in range(len(coord_layers[i])):
                for k in range(len(coord_layers[i+1])):
                    if random.random() < 0.05:
                        x1, y1 = coord_layers[i][j]
                        x2, y2 = coord_layers[i+1][k]
                        self.lines.append(self.visual.create_line(x1+radius/2,y1+radius/2,x2+radius/2,y2+radius/2))
        for x, y in c:
            self.neurons.append(self.visual.create_oval(x,y,x+radius,y+radius,width=0,fill="#f00"))

    def update_brain(self):
        if not self.chosen_one:
            return
        inputs = self.agent.local_Q.forward_visual_mode(self.chosen_one.give_state())
        k = 0
        for i in inputs:
            for num in range(len(i)):
                value = i[num].item()#-mn)/(mx-mn)
                r = int(255*(1-value))
                g = int(255*value)
                color = "#" + "".join([format(val, '02X') for val in (r,g,0)])
                neuron = self.neurons[k]
                self.visual.itemconfig(neuron, fill=color)
                k += 1
        
    def on_click(self, event):
        x = int(event.x // self.rectangle_size)
        y = int(event.y // self.rectangle_size)
        if self.env.board[y,x] > 0:
            for snake in self.env.snakes:
                if snake.id == self.env.board[y,x]:
                    self.chosen_one = snake
            random.seed(snake)
            color = get_color()
            self.visual.configure(background=color)

    def key_down(self, event):
        key = repr(event.char)

        if self.chosen_one and self.controller:
            if key in self.commands: 
                self.chosen_one.direction = CONTROLLER_MAPPING[self.chosen_one.direction][self.commands[key]]
        if key == "'q'":
            self.root.quit()
        if key == "'c'":
            self.controller = not self.controller
        if key == "'z'":
            self.speed += 0.005
        if key == "'x'":
            self.speed -= 0.005
        if key == "'p'":
            self.pause = not self.pause


env = Environment(row=25, col=25, num_snakes=6, throw_food_every=30)
gui = GameGrid(env)