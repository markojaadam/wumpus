#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np

class Environment(object):
    def __init__(self):
        self.matrix = np.zeros([4,4,5])
        # Első két D: koordináta a térképen, harmadik D: mi van a térkép cellájában?
        # [szörny (0/1), szakadék (0/1), bűz (0/1), szellő (0/1), arany (0/1)]
        self.wumpus_loc = np.array([[0,2]])
        self.pit_locs = np.array([[2,0],[2,2],[3,3]])
        self.gold_loc = np.array([[1,2]])

        self.stench_locs = []
        for loc in [[0,1],[1,0],[0,-1],[-1,0]]:
            new_loc = self.wumpus_loc[0]+loc
            if not (np.sum(new_loc < 0) or sum(new_loc >= self.matrix.shape[:2])):
                if new_loc.tolist() not in self.stench_locs:
                    self.stench_locs.append(new_loc.tolist())
        self.stench_locs = np.asarray(self.stench_locs)

        self.breeze_locs = []
        for pit_loc in self.pit_locs:
            for loc in [[0,1],[1,0],[0,-1],[-1,0]]:
                new_loc = pit_loc+loc
                if not (np.sum(new_loc < 0) or sum(new_loc >= self.matrix.shape[:2])):
                    if new_loc.tolist() not in self.breeze_locs:
                        self.breeze_locs.append(new_loc.tolist())
        self.breeze_locs = np.asarray(self.breeze_locs)

        self.place_objects()
        self.start_matrix = self.matrix.copy()

    def place_objects(self):
        for pos,loc in enumerate([self.wumpus_loc,self.pit_locs,self.stench_locs,
                              self.breeze_locs,self.gold_loc]):
            for i,j in loc:
                self.matrix[i][j][pos] = 1

    def kill_wumpus(self):
        # Eltávolítja a szörnyet a térképről és törli a bűz-jeleket
        self.matrix[self.wumpus_loc[0,0]][self.wumpus_loc[0,1]][0] = 0 # Wumpus törölve
        for i,j in self.stench_locs:
            self.matrix[i][j][2] = 0 # Bűz törölve

    def reset_matrix(self):
        # Visszaállítja a térképet az eredeti állapotába
        self.matrix = self.start_matrix.copy()


class Agent(object):
    def __init__(self, Environment, learning_rate=0.5, discount_factor=0.8, debug=False):
        self.env = Environment
        self.start_matrix = self.env.start_matrix.copy()
        self.debug = debug
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.step, self.history, self.action_history = 0, [], []
        self.state = np.array([0,0,1,0]) # Első két bool: x és y koord, 3. van nyílvesszőnk, 4.: szörny kilőve
        self.arrowpos = self.state[:2].copy()
        self.map = np.zeros(self.env.matrix.shape)
        self.qtable = np.zeros(self.env.matrix.shape[:2]+(8,)) # A térkép minden pontján minden cselekvésre értéket tárol
        self.rewards = np.zeros(self.env.matrix.shape[:2]+(8,)) # A térkép minden pontján minden cselekvésre értéket tárol
        self.dir_dic = {'left':[-1,0],'right':[1,0],'down':[0,-1],'up':[0,1]} # Irányok kódolása
        actions = ['move_' + d for d in self.dir_dic.keys()] + ['shoot_' + d for d in self.dir_dic.keys()]
        self.action_map = dict([(j,i) for (i,j) in enumerate(actions)]) # Akciók kódolása
        self.wins, self.deaths = 0,0

    def eval_actions(self):
        '''Meggátolja, hogy az ágens kimenjen a mátrixból,
        továbbá csak akkor engedi lőni, ha bűzt érez.'''
        available_actions = []
        for d in self.dir_dic.keys():
            newpos = self.state[:2] + self.dir_dic[d]
            if not (np.sum(newpos < 0) or np.sum(newpos >= self.env.matrix.shape[:2])):
                available_actions.append('move_' + d)
                if self.state[:2] in self.env.stench_locs and self.state[2] == 1:
                    available_actions.append('shoot_' + d)
        return(available_actions)

    def take_action(self,action):
        '''Dekódolja a stringként beadott akciót és tárolja az akciótörténetben'''
        self.step += 1
        act,direction = action.split('_')
        if act == 'move':
            if self.debug:
                q_value = self.qtable[tuple(self.state[:2])][self.action_map[action]]
                print('Move %s on %s (%s)'%(direction,self.state[:2],q_value))
            result = self.move(direction)
        elif act == 'shoot':
            if self.debug:
                q_value = self.qtable[tuple(self.state[:2])][self.action_map[action]]
                print('Shoot %s on %s %s'%(direction,self.state[:2],q_value))
            result = self.shoot(direction)
        self.action_history.append(action)
        return(result)

    def reset_state(self):
        '''Visszaállítja az ágens állapotát és a térképet'''
        self.state = np.array([0,0,1,0])
        self.arrowpos = self.state[:2].copy()
        self.env.reset_matrix()

    def reset_knowlegde(self):
        self.qtable = np.zeros(self.env.matrix.shape[:2]+(8,))
        self.rewards = np.zeros(self.env.matrix.shape[:2]+(8,))

    def move(self,direction):
        '''Az ágens elmozdul a megadott irányba a mátrixon belül'''
        oldpos = self.state[:2].copy()
        action_ix = self.action_map['move_'+direction]
        self.state[:2] += self.dir_dic[direction]
        obj = self.env.matrix[tuple(self.state[:2])]
        self.map[tuple(self.state[:2])] = self.env.matrix[tuple(self.state[:2])]
        if obj[0] == 1 or obj[1] == 1: # szörnyre, vagy szakadékba lép: -10
            result = 'You died!'
            if self.debug: print('You died on %s.'%self.state[:2])
            self.history.append(-10)
            self.reset_state()
            self.rewards[(tuple(oldpos))][action_ix] = -10
            self.deaths += 1
        elif obj[4] == 1: # Aranyat talál: +100
            result = 'Gold found!'
            if self.debug: print('You won on %s.'%self.state[:2])
            self.history.append(100)
            self.reset_state()
            self.rewards[(tuple(oldpos))][action_ix] = 100
            self.wins += 1
        else: # Nem történik semmi: -1
            result = 'nope'
            self.history.append(-1)
            self.rewards[(tuple(oldpos))][action_ix] = -1
        self.update_qtable(oldpos,self.state[:2],action_ix)
        self.update_arrowpos()
        return(result)

    def shoot(self,direction):
        '''Kilövi a nyilat a megadott irányba'''
        action_ix = self.action_map['shoot_'+direction]
        self.state[2] = 0 # Nyíl kilőve
        wumpus_died = False
        while not (np.sum(self.arrowpos < 0) or sum(self.arrowpos >= self.env.matrix.shape[:2])):
            if (self.arrowpos == self.env.wumpus_loc[0]).all(): # A szörny meghalt: +10
                result = 'Wumpus died!'
                if self.debug: print('Wumpus died.')
                self.state[3] = 1
                self.env.kill_wumpus()
                self.rewards[(tuple(self.state[:2]))][action_ix] = 10
                self.history.append(10)
                wumpus_died = True
            self.arrowpos += self.dir_dic[direction]
        if not wumpus_died: # Kilőttük a nyilat, de nem történt semmi
            result = 'nope'
            if self.debug: print('Nothing happened.')
            self.history.append(-1)
            self.rewards[(tuple(self.state[:2]))][action_ix] = -1
        self.update_qtable(self.state[:2],self.state[:2],action_ix)
        self.arrowpos = None
        return(result)

    def update_arrowpos(self):
        if self.state[2]:
            self.arrowpos = self.state[:2].copy()

    def update_qtable(self, state, next_state, action_ix):
        '''Frissíti az ágens Q-tábláját'''
        state, next_state = tuple(state), tuple(next_state)
        r = self.rewards[state][action_ix]
        q = self.qtable[state][action_ix]
        new_q = q + self.learning_rate * (r + self.discount_factor * max(self.qtable[next_state]) - q)
        self.qtable[state][action_ix] = new_q
        if np.max(np.abs(self.qtable[state])) > 0:
            self.qtable[state] = self.qtable[state]/np.max(np.abs(self.qtable[state]))

    def get_best_action(self,default = 0.01):
        '''Kiválasztja a Q tábla alapján a legjobb lépést'''
        best = np.random.choice(self.eval_actions())
        if not np.random.uniform(0,1) < default:
            available_moves = np.array([self.action_map[a] for a in self.eval_actions()])
            if not (self.qtable[tuple(self.state[:2])][available_moves] == 0).all():
                max_q_loc = np.where(self.qtable[tuple(self.state[:2])] == \
                                     max(self.qtable[tuple(self.state[:2])][available_moves]))
                max_q_loc = np.intersect1d(available_moves,max_q_loc[0])
                if len(max_q_loc) > 1:
                    action_ix = np.random.choice(max_q_loc)
                else:
                    action_ix = max_q_loc
                for action, ix in self.action_map.items():
                    if ix == action_ix:
                        best = action
        return(best)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_discount_factor(self, discount_factor):
        self.discount_factor = discount_factor

class GUI(object):
    def __init__(self):
        import Tkinter as tk
        import tkMessageBox
        import ttk
        import time

        self.tk = tk
        self.ttk = ttk
        self.tkMessageBox = tkMessageBox

        self.root = tk.Tk()
        self.root.title('Q learning in Wumpus World')

        self.env = Environment()
        self.agent = Agent(self.env)
        self.stopmode = False
        self.exitcmd = False

        self._env_text = dict(enumerate(list('WPSBG')))
        self._env_grid = dict(zip(range(self.env.matrix.shape[-1]),[0, 1, 3, 5, 2]))
        self._env_colors = dict(zip(range(self.env.matrix.shape[-1]), ['red','black','grey','grey','gold']))

        self.mainframe = tk.Frame(master=self.root)
        self.sideframe = tk.Frame(master=self.root)
        self.mainframe.grid(row=0, column=0)
        self.sideframe.grid(row=0, column=1)
        self.messagebox = tk.Text(self.sideframe, height=1, width=20)
        self.iternum = tk.Spinbox(self.sideframe, width=10, values =  list(range(100,1001,100))+list(range(2000,10001,1000)))
        self.learnbox = tk.Spinbox(self.sideframe, width=10, values= tuple([i/10 for i in range(1,11)]))
        self.discbox= tk.Spinbox(self.sideframe, width=10, values=tuple([i/10 for i in range(1,11)]))
        self.startbtn = ttk.Button(master=self.sideframe, text='Start', command=self.start_run)
        self.stopbtn = ttk.Button(master=self.sideframe, text='Stop', command=self.stop)
        self.exitbtn = ttk.Button(master=self.sideframe, text='Exit', command=self.exit)

        ttk.Label(self.sideframe, text="Learning Rate: ", padding=10).grid(row=0, column=0, sticky='e')
        self.learnbox.grid(row=0, column=1, sticky='w')
        ttk.Label(self.sideframe, text="Discount Factor: ", padding=10).grid(row=1, column=0, sticky='e')
        self.discbox.grid(row=1, column=1, sticky='w')
        ttk.Label(self.sideframe, text="Number of iterations: ", justify='right', padding=10).grid(row=2, column = 0, sticky='e')
        self.iternum.grid(row=2, column=1, padx=10, sticky='w')
        ttk.Label(self.sideframe, text="Message: ", justify='right', padding=10).grid(row=3, column = 0, sticky='e')
        self.messagebox.grid(row=3, column=1, padx=10, sticky='w')
        self.startbtn.grid(row=4,column=0, padx=10)
        self.stopbtn.grid(row=4, column=1, padx=10)
        self.exitbtn.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        self.world = tk.Canvas(self.mainframe, width=350, height=350)
        self.q_move = tk.Canvas(self.mainframe, width=250, height=250)
        self.q_shoot = tk.Canvas(self.mainframe, width=250, height=250)

        ttk.Label(self.mainframe, text="Wumpus World", justify='center', padding=10).grid(row=0, column=0, columnspan=2)
        self.world.grid(row=1, column=0, columnspan=2)
        ttk.Label(self.mainframe, text="Q value for movements", justify='center', padding=10).grid(row=2,column=0)
        self.q_move.grid(row=3, column=0, padx=(0, 5))
        ttk.Label(self.mainframe, text="Q value for shoots", justify='center', padding=10).grid(row=2,column=1)
        self.q_shoot.grid(row=3, column=1, padx=(5, 0))
        for canvas in [self.world,self.q_move,self.q_shoot]:
            w,h = int(canvas['width']), int(canvas['height'])
            for i in range(4):
                for j in range(4):
                    canvas.create_rectangle(w*j/4, h*i/4, w*j/4 + w/4, h*i/4 + h/4, fill='white')


        def _create_circle(self, x, y, r, **kwargs):
            return self.create_oval(x - r, y - r, x + r, y + r, **kwargs)

        tk.Canvas.create_circle = _create_circle
        self.world_objs = self.draw_world()
        self.q_m_objs, self.q_s_objs = self.draw_qvalues()

    def coord_on_canvas(self, canvas, x, y, gridpos=4):
        grid_dic = dict(zip(range(9),[(i / 4, j / 4) for i, j in zip([1, 2, 3] * 3, [1] * 3 + [2] * 3 + [3] * 3)]))
        xg,yg = grid_dic[gridpos]
        w, h = int(canvas['width']), int(canvas['height'])
        x0, y0 = w*x/4*(1-xg) + w*(x+1)/4*(xg), h - (h*y/4*(yg) + h*(y+1)/4*(1-yg))
        return(x0, y0)

    def to_coords(self,array_tuple):
        return(np.array([a.tolist() for a in array_tuple]).T)

    def draw_world(self):
        world_objs = []
        for i in range(self.env.matrix.shape[-1]):
            coord_pairs = self.to_coords(np.where(self.env.matrix[:, :, i] == 1))
            for x, y in coord_pairs:
                x0, y0 = self.coord_on_canvas(self.world, x, y, self._env_grid[i])
                world_objs.append(self.world.create_text(x0,y0,text=self._env_text[i],fill=self._env_colors[i],
                                                         font=('Helvetica', '14')))
        return(world_objs)

    def draw_agent(self):
        x,y = self.agent.state[:2]
        x0, y0 = self.coord_on_canvas(self.world, x, y)
        return(self.world.create_circle(x0,y0,10,fill='green'))

    def draw_arrow(self):
        x,y = self.agent.arrowpos
        x0, y0 = self.coord_on_canvas(self.world, x, y, 1)
        return(self.world.create_circle(x0,y0,5,fill='red'))

    def draw_qvalues(self):
        grid_dic = dict(zip([3, 0, 1, 2, 7, 4, 5, 6], [3,7,5,1] * 2))
        q_m_objs,q_s_objs = [], []
        for i in range(self.agent.qtable.shape[-1]):
            qtable = self.agent.qtable[:,:,i]
            target_canvas = self.q_move if i < 4 else self.q_shoot
            for x in range(qtable.shape[0]):
                for y in range(qtable.shape[1]):
                    q = qtable[x,y]
                    text = '%.2f'%q if q < 1000 else '%.0f'%q
                    c = 'green' if q > 0 else 'red' if q < 0 else 'black'
                    x0, y0 = self.coord_on_canvas(target_canvas, x, y, grid_dic[i])
                    if i < 4:
                        q_m_objs.append(target_canvas.create_text(x0,y0, text=text, fill=c, font=('Helvetica', '8')))
                    else:
                        q_s_objs.append(target_canvas.create_text(x0, y0, text=text, fill=c, font=('Helvetica', '8')))
        return(q_m_objs,q_s_objs)


    def start_run(self):
        self.stopmode = False
        self.agent.reset_state()
        self.agent.reset_knowlegde()
        self.agent.set_learning_rate(float(self.learnbox.get()))
        self.agent.set_discount_factor(float(self.discbox.get()))
        self.clear_world()
        self.clear_qtables()
        self.world_objs = self.draw_world()
        self.q_m_objs, self.q_s_objs = self.draw_qvalues()
        agent_circle = self.draw_agent()
        arrow_circle = self.draw_arrow()
        self.world.update()
        self.world.after(1000)
        for i in range(int(self.iternum.get())):
            if self.stopmode:
                break
            self.world.after(10)
            a = self.agent.get_best_action()
            result = self.agent.take_action(a)
            self.clear_world()
            self.clear_qtables()
            self.world.delete(agent_circle)
            self.world.delete(arrow_circle)
            self.world_objs = self.draw_world()
            self.q_m_objs, self.q_s_objs = self.draw_qvalues()
            agent_circle = self.draw_agent()
            if self.agent.arrowpos is not None:
                arrow_circle = self.draw_arrow()
            self.messagebox.delete('1.0', 'end')
            self.messagebox.insert('1.0',result)
            self.world.update()
            if result != 'nope':
                self.world.after(490)
        if not self.stopmode:
            self.tkMessageBox.showinfo(message='Exection ended')
        if not self.exitcmd:
            self.clear_world()
            self.world.delete(agent_circle)
            self.world.delete(arrow_circle)
            self.messagebox.delete('0.1','end')


    def clear_world(self):
        for wo in self.world_objs:
            self.world.delete(wo)

    def clear_qtables(self):
        for qmo in self.q_m_objs:
            self.q_move.delete(qmo)
        for qso in self.q_s_objs:
            self.q_shoot.delete(qso)

    def stop(self):
        self.stopmode = True

    def exit(self):
        self.exitcmd = True
        self.stop()
        self.root.destroy()

    def main_loop(self):
        self.root.mainloop()

if __name__ == '__main__':
    gui = GUI()
    gui.main_loop()
