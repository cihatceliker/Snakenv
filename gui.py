from snake import CONTROLLER_MAPPING, POSSIBLE_DIRECTIONS_TO_GO, EMPTY, FOOD, EGG, REMAINS, dirs, Snake
from tkinter import Frame, Canvas, Tk
import time
import threading
import pickle
from environment import Environment
import numpy as np
import random
from pyscreenshot import grab
from model_translation import NumpyAgent

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
    noise = 100
    r = min(255, max(0, 255-r+random.randint(-noise, noise)))
    g = min(255, max(0, 255-g+random.randint(-noise, noise)))
    b = min(255, max(0, 255-b+random.randint(-noise, noise)))
    r = hex(r).split('x')[-1]
    g = hex(g).split('x')[-1]
    b = hex(b).split('x')[-1]
    return "#"+r+g+b


class GameGrid():

    def __init__(self, speed=0.05, size=720):
        self.root = Tk()
        self.root.configure(background=BACKGROUND_COLOR)
        self.left_frame = Frame(self.root, width=size, height=size, bg=BACKGROUND_COLOR)
        self.left_frame.grid(row=0, column=0)
        self.right_frame = Frame(self.root, width=size/2, height=size, bg=BACKGROUND_COLOR)
        self.right_frame.grid(row=0, column=1)
        self.game = Canvas(self.left_frame, width=size, height=size, bg=BACKGROUND_COLOR)
        self.game.pack()
        self.visual = Canvas(self.right_frame, width=size/2, height=size, bg=BACKGROUND_COLOR)
        self.visual.pack()

        self.agent = pickle.load(open("numpy_brain.snk","rb"))
        self.env = Environment(row=20, col=20, num_snakes=4, throw_food_every=20)
        #self.env = Environment(row=60, col=60, num_snakes=20, throw_food_every=8)
        self.env.reset()
        self.speed = speed
        self.size = size
        self.grid_padding = 10
        self.rectangle_size = size/self.env.row
        self.pause = False
        self.chosen_one = None
        self.controller = False
        self.take_ss = False
        self.quit = False
        self.image_counter = 0

        self.init_board()
        self.init_brain()

        self.root.title('Snakenv')
        self.commands = {"'w'": 0,"'a'": 3,"'s'": 2,"'d'": 1}
        self.root.bind("<Key>", self.key_down)
        self.root.bind('<Button-1>', self.on_click)
        self.run_game()
        # to make it more responsive
        threading.Thread(target=self.key_down).start()

    def run_game(self):
        obs = self.env.reset()
        score = 0
        while not self.quit:
            if not self.pause:
                action_list = []
                for state, reward, _, _ in obs:
                    action = self.agent.select_action(state)
                    action_list.append(action)
                obs_ = self.env.step(action_list, controlled=None if not self.chosen_one or not self.controller else self.chosen_one.id)
                obs = obs_
                if self.chosen_one:
                    for _,_,_,info in obs:
                        id_, info = info
                        if info != "" and id_ == self.chosen_one.id:
                            print(info)
                # if chosen snake dies
                if self.chosen_one not in self.env.snakes:
                    self.chosen_one = None
                    self.controller = False
                    self.visual.configure(background=BACKGROUND_COLOR)
                self.update_board()
                self.update_brain()
                time.sleep(max(self.speed, 0))
            self.root.update()
        print("done", score)

    def update_board(self):
        for i in range(self.env.row):
            for j in range(self.env.col):
                rect, _, fillers = self.game_area[i][j]
                curr = int(self.env.board[i, j])
                if curr in COLORS:
                    color = COLORS[curr]
                    self.game.itemconfig(rect, fill=color)

                    colors = [COLORS[EMPTY]] * 4

                    for m in range(4):
                        x_, y_ = dirs[m]
                        cur_dir = Snake.check_wall_hit((i+x_, j+y_), self.env.row, self.env.col)
                        if self.env.board[cur_dir] == curr:
                            colors[m] = color
                    for c in range(4):
                        self.game.itemconfig(fillers[c], fill=colors[c])
                else:
                    color = COLORS[EMPTY] # smthing effects random
                    for snake in self.env.snakes:
                        if curr != snake.id: continue
                        if (i, j) == snake.head:
                            color = "black"
                            if self.chosen_one and curr == self.chosen_one.id:
                                color = "white"
                        else:
                            # comment seed to see the rainbow mode
                            random.seed(snake.id*17)
                            color = get_color()
                        # fillers for snake body parts
                        for k in range(len(snake.body)):
                            if snake.body[k] == (i, j):
                                colors = [COLORS[EMPTY]] * 4
                                next_body_part = None
                                prev_body_part = None
                                if k + 1 != len(snake.body):
                                    next_body_part = snake.body[k+1]
                                if k != 0:
                                    prev_body_part = snake.body[k-1]
                                for m in range(4):
                                    x_, y_ = dirs[m]
                                    cur_dir = Snake.check_wall_hit((i+x_, j+y_), self.env.row, self.env.col)
                                    if self.env.board[cur_dir] == curr and \
                                        (prev_body_part == cur_dir or next_body_part == cur_dir):
                                        colors[m] = color
                                for c in range(4):
                                    self.game.itemconfig(fillers[c], fill=colors[c])
                                break
                        self.game.itemconfig(rect, fill=color)


    def init_board(self):
        color = COLORS[EMPTY]
        def draw(x1, y1, w, h, func, color=color):
            return func(x1, y1, x1+w, y1+h, fill=color, width=0)
        
        rect = draw(0, 0, self.size, self.size, self.game.create_rectangle)
        
        self.game_area = []
        # rectangle, oval, fillers_tuple
        
        for i in range(self.env.row):
            row = []
            for j in range(self.env.col):
                # create_oval for food ???
                fillers = {}

                x = j * self.rectangle_size + self.grid_padding
                y = i * self.rectangle_size + self.grid_padding
                sz = self.rectangle_size - 2 * self.grid_padding

                fillers[0] = draw(x, y - self.grid_padding, sz, self.grid_padding, self.game.create_rectangle)
                fillers[1] = draw(x + sz, y, self.grid_padding, sz, self.game.create_rectangle)
                fillers[2] = draw(x, y + sz, sz, self.grid_padding, self.game.create_rectangle)
                fillers[3] = draw(x - self.grid_padding, y, self.grid_padding, sz, self.game.create_rectangle)

                #oval = draw(x, y, sz, sz, self.game.create_oval)
                rect = draw(x, y, sz, sz, self.game.create_rectangle)
                #oval = draw(x-self.grid_padding, y-self.grid_padding, self.rectangle_size, self.rectangle_size, self.game.create_oval)
                row.append((rect, None, fillers))

            self.game_area.append(row)

    def init_brain(self, radius=10, padding=1):
        # this method draws all the neurons and the lines for the visuals.    
        layers = [self.env.observation_space, 64, 64, self.env.action_space]
        num_layers = len(layers)
        offset_x = []
        for i in range(num_layers):
            num_neurons = layers[i]
            start = (self.size-num_neurons*radius-(num_neurons-1)*padding)/2
            offset_x.append(start)
        layer_size = self.size/2 / num_layers
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
        self.neurons = []
        self.lines = []
        coord_layers = []
        k = 0
        for i in layers:
            coord_layers.append([])
            for _ in range(i):
                coord_layers[-1].append(c[k])
                k+=1
        num_neurons = 0
        for i in range(len(coord_layers)-1):
            num_neurons += len(coord_layers[i])
            for j in range(len(coord_layers[i])):
                n = 0
                for k in range(len(coord_layers[i+1])):
                    # randomized because drawing every line is ugly and very slow
                    if (i == len(coord_layers)-2 and random.random() < 0.2) or random.random() < 0.03:
                        x1, y1 = coord_layers[i][j]
                        x2, y2 = coord_layers[i+1][k]
                        self.lines.append((self.visual.create_line(x1+radius/2,y1+radius/2,x2+radius/2,y2+radius/2), num_neurons+n))
                    n+=1
        for x, y in c:
            self.neurons.append(self.visual.create_oval(x,y,x+radius,y+radius,width=0,fill=BACKGROUND_COLOR))
    
    def update_brain(self):
        if not self.chosen_one:
            for neuron in self.neurons:
                self.visual.itemconfig(neuron, fill=BACKGROUND_COLOR)
            for line, _ in self.lines:
                self.visual.itemconfig(line, fill=BACKGROUND_COLOR)
            return
        inputs = self.agent.forward_visual_mode(self.chosen_one.give_state())
        k = 0
        for i in inputs:
            for num in range(len(i)):
                p = i[num].item()
                r = int((1.0-p) * 255 + 0.5)
                g = int(p * 255 + 0.5)
                color = "#" + "".join([format(val, '02X') for val in (r,g,0)])
                neuron = self.neurons[k]
                self.visual.itemconfig(neuron, fill=color)
                # making neuron connections colorful. not necessary
                """
                for line in range(len(self.lines)):
                    if self.lines[line] is None: continue
                    if self.lines[line][1] == k:
                        self.visual.itemconfig(self.lines[line][0], fill=color)
                """
                k += 1
        # p represents the chosen snakes' hunger. more hunger, more red.
        p = inputs[0][-1]
        g = int((1.0-p) * 160 + 0.5)
        r = int(p * 160 + 0.5)
        color = "#"+"".join([format(val, '02X') for val in (r,g,0)])
        for line, _ in self.lines:
            if np.random.random() < 0.5:
                self.visual.itemconfig(line, fill=color)
            else:
                self.visual.itemconfig(line, fill=BACKGROUND_COLOR)
        self.visual.configure(background=color)
        self.take_screenshot()

    def take_screenshot(self):
        # game windows should be on the left bottom corner
        if self.take_ss:
            x = 1
            y = 359
            img = grab(bbox=(x,y,x+1077,y+718))
            img.save("ss/ss"+str(self.image_counter)+".png")
            self.image_counter += 1
        
    def on_click(self, event):
        self.chosen_one = None
        x = int(event.x // self.rectangle_size)
        y = int(event.y // self.rectangle_size)
        if self.env.board[y,x] > 0:
            for snake in self.env.snakes:
                if snake.id == self.env.board[y,x]:
                    self.chosen_one = snake
        else:
            self.controller = False

    def key_down(self, event):
        key = repr(event.char)
        if self.chosen_one and self.controller:
            if key in self.commands: 
                self.chosen_one.direction = CONTROLLER_MAPPING[self.chosen_one.direction][self.commands[key]]
        if key == "'q'":
            self.quit = True
        if key == "'z'":
            self.speed = min(0.1, self.speed+0.025)
        if key == "'x'":
            self.speed = max(0, self.speed-0.025)
        if key == "'p'":
            self.pause = not self.pause
        if key == "'c'":
            self.controller = not self.controller
        if key == "'m'":
            self.take_ss = not self.take_ss


if __name__ == "__main__":
    GameGrid()
