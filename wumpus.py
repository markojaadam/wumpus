#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
pyver = sys.version_info.major
if pyver > 2:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
else:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import colors as mcolors

class Environment(object):
    def __init__(self, rownum=4, colnum=4, default = False):
        # Első két D: koordináta a térképen, harmadik D: mi van a térkép cellájában?
        # [szörny (0/1), szakadék (0/1), bűz (0/1), szellő (0/1), arany (0/1)]
        exceptions = np.array([[0,0],[0,1],[1,0],[1,1]])
        if default:
            self.rownum = 4
            self.colnum = 4
            self.matrix = np.zeros([self.rownum, self.colnum, 5])
            self.wumpus_locs = np.array([[0,2]])
            self.pit_locs = np.array([[2,0],[2,2],[3,3]])
            self.gold_locs = np.array([[1,2]])
        else:
            self.rownum = rownum
            self.colnum = colnum
            self.matrix = np.zeros([self.rownum, self.colnum, 5])
            self.gold_locs = self.generate_locations(1,exceptions)
            exceptions = np.concatenate([self.gold_locs,exceptions])
            self.wumpus_locs = self.generate_locations(int(self.rownum/2),exceptions)

        self.stench_locs = []
        for wumpus_loc in self.wumpus_locs:
            for loc in [[0,1],[1,0],[0,-1],[-1,0]]:
                new_loc = wumpus_loc+loc
                if not (np.sum(new_loc < 0) or sum(new_loc >= self.matrix.shape[:2])):
                    if new_loc.tolist() not in self.stench_locs:
                        self.stench_locs.append(new_loc.tolist())
        self.stench_locs = np.asarray(self.stench_locs)

        if not default:
            exceptions = np.concatenate([exceptions, self.wumpus_locs, self.stench_locs])
            self.pit_locs = self.generate_locations(3, exceptions)

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
        for pos,loc in enumerate([self.wumpus_locs,self.pit_locs,self.stench_locs,
                              self.breeze_locs,self.gold_locs]):
            for i,j in loc:
                self.matrix[i][j][pos] = 1

    def kill_wumpus(self,pos):
        # Eltávolítja a szörnyet a térképről és törli a bűz-jeleket
        self.matrix[tuple(pos)][0] = 0
        for loc in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
            if not (np.sum(pos + loc < 0) or sum(pos + loc >= self.matrix.shape[:2])):
                self.matrix[tuple(pos + loc)][2] = 0

    def reset_matrix(self):
        # Visszaállítja a térképet az eredeti állapotába
        self.matrix = self.start_matrix.copy()

    def generate_locations(self,n_loc,exceptions):
        exceptions = np.asarray(exceptions)
        locations = []
        for i in range(n_loc):
            while True:
                x, y = np.random.randint(self.rownum), np.random.randint(self.colnum)
                if not ([x,y] == exceptions).all(1).any() and [x,y] not in locations:
                    locations.append([x,y])
                    break
        return(np.array(locations))




class Agent(object):
    def __init__(self, Environment, learning_rate=0.5, discount_factor=0.8, random_factor=0.01,
                 reward_dic = None, debug=False):
        default_r = [('win',100),('wumpus_killed',10),('death',-10),('move_nothing',-1),('shoot_nothing',-5)]
        self.reward_dic = dict(default_r )if not reward_dic else reward_dic if type(reward_dic) == dict else dict(reward_dic)
        self.env = Environment
        self.start_matrix = self.env.start_matrix.copy()
        self.debug = debug
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.random_factor = random_factor
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

    def eval_actions(self, state):
        '''Meggátolja, hogy az ágens kimenjen a mátrixból,
        továbbá csak akkor engedi lőni, ha bűzt érez.'''
        available_actions = []
        for d in self.dir_dic.keys():
            newpos = state[:2] + self.dir_dic[d]
            if not (np.sum(newpos < 0) or np.sum(newpos >= self.env.matrix.shape[:2])):
                available_actions.append('move_' + d)
                if state[2] == 1:
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
        self.step, self.history, self.action_history = 0, [], []
        self.wins, self.deaths = 0, 0

    def move(self,direction):
        '''Az ágens elmozdul a megadott irányba a mátrixon belül'''
        oldstate = self.state.copy()
        action_ix = self.action_map['move_'+direction]
        self.state[:2] += self.dir_dic[direction]
        obj = self.env.matrix[tuple(self.state[:2])]
        self.map[tuple(self.state[:2])] = self.env.matrix[tuple(self.state[:2])]
        if obj[0] == 1 or obj[1] == 1: # szörnyre, vagy szakadékba lép: -10
            result = 'You died!'
            if self.debug: print('You died on %s.'%self.state[:2])
            self.history.append(self.reward_dic['death'])
            self.reset_state()
            self.rewards[(tuple(oldstate[:2]))][action_ix] = self.reward_dic['death']
            self.deaths += 1
        elif obj[4] == 1: # Aranyat talál: +100
            result = 'Gold found!'
            if self.debug: print('You won on %s.'%self.state[:2])
            self.history.append(self.reward_dic['win'])
            self.reset_state()
            self.rewards[(tuple(oldstate[:2]))][action_ix] = self.reward_dic['win']
            self.wins += 1
        else: # Nem történik semmi: -1
            result = 'nope'
            self.history.append(self.reward_dic['move_nothing'])
            self.rewards[(tuple(oldstate[:2])),action_ix] = self.reward_dic['move_nothing']
        self.update_qtable(oldstate,self.state,action_ix)
        self.update_arrowpos()
        return(result)

    def shoot(self,direction):
        '''Kilövi a nyilat a megadott irányba'''
        oldstate = self.state.copy()
        action_ix = self.action_map['shoot_'+direction]
        self.state[2] = 0 # Nyíl kilőve
        self.arrowpos += self.dir_dic[direction]
        if (self.arrowpos == self.env.wumpus_locs).all(1).any(): # A szörny meghalt: +10
            result = 'Wumpus died!'
            if self.debug: print('Wumpus died.')
            self.state[3] = 1
            self.env.kill_wumpus(self.arrowpos)
            self.rewards[(tuple(oldstate[:2]))][action_ix] = self.reward_dic['wumpus_killed']
            self.history.append(self.reward_dic['wumpus_killed'])
        else: # Kilőttük a nyilat, de nem történt semmi
            result = 'nope'
            if self.debug: print('Nothing happened.')
            self.history.append(self.reward_dic['shoot_nothing'])
            self.rewards[(tuple(self.state[:2]))][action_ix] = self.reward_dic['shoot_nothing']
        self.update_qtable(oldstate,self.state,action_ix)
        self.arrowpos = None
        return(result)

    def update_arrowpos(self):
        if self.state[2]:
            self.arrowpos = self.state[:2].copy()

    def update_qtable(self, state, next_state, action_ix):
        '''Frissíti az ágens Q-tábláját'''
        pos, next_pos = tuple(state[:2]), tuple(next_state[:2])
        r = self.rewards[pos][action_ix]
        q = self.qtable[pos][action_ix]
        available_actions = np.array([self.action_map[a] for a in self.eval_actions(self.state)])
        qmax = np.max(self.qtable[next_pos][available_actions ])
        new_q = q + self.learning_rate * (r + self.discount_factor * qmax - q)
        self.qtable[pos][action_ix] = new_q

    def get_best_action(self):
        '''Kiválasztja a Q tábla alapján a legjobb lépést'''
        best = np.random.choice(self.eval_actions(self.state))
        if not np.random.uniform(0,1) < self.random_factor:
            available_actions = np.array([self.action_map[a] for a in self.eval_actions(self.state)])
            if not (self.qtable[tuple(self.state[:2])][available_actions] == 0).all():
                max_q_loc = np.where(self.qtable[tuple(self.state[:2])] == \
                                     max(self.qtable[tuple(self.state[:2])][available_actions]))
                max_q_loc = np.intersect1d(available_actions,max_q_loc[0])
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

    def set_random_factor(self,random_factor):
        self.random_factor = random_factor

class GUI(object):
    def __init__(self, rownum=6, colnum=6, env_default=False, reward_params=None):
        if pyver > 2:
            import tkinter as tk
            import tkinter.ttk as ttk
            import tkinter.messagebox as tkMessageBox
        else:
            import Tkinter as tk
            import tkMessageBox
            import ttk

        self.tk = tk
        self.ttk = ttk
        self.tkMessageBox = tkMessageBox

        self.root = tk.Tk()
        self.root.protocol('WM_DELETE_WINDOW', self.exit)
        self.root.title('Q learning in Wumpus World')

        self.rownum = rownum
        self.colnum = colnum
        self.env_default = env_default
        self.learning_rate = tk.StringVar(self.root)
        self.discount_factor = tk.StringVar(self.root)
        self.random_factor = tk.StringVar(self.root)
        self.wins = tk.StringVar(self.root)
        self.deaths = tk.StringVar(self.root)
        self.sumreward = tk.StringVar(self.root)
        self.pause_on_evet = tk.BooleanVar(self.root)
        self.env = Environment(self.rownum,self.colnum, default = self.env_default)
        self.rownum, self.colnum = self.env.rownum, self.env.colnum # Override if Environment set to default mode
        self.agent = Agent(self.env, reward_dic=reward_params)
        self.learning_rate.set(str(self.agent.learning_rate))
        self.discount_factor.set(str(self.agent.discount_factor))
        self.random_factor.set(str(self.agent.random_factor))
        for stringvar in [self.wins, self.deaths, self.sumreward]:
            stringvar.set('0')
        self.stopmode = False
        self.run_no, self.lines, self.plot_history = 0, [], []
        if matplotlib.__version__ >= '2.0':
            self._colors = list(mcolors.BASE_COLORS)
        else:
            self._colors = ['c', 'b', 'w', 'g', 'y', 'k', 'r', 'm']
        self._colors.remove('w')
        np.random.shuffle(self._colors)

        self._env_text = dict(enumerate(list('WPSBG')))
        self._env_grid = dict(zip(range(self.env.matrix.shape[-1]),[0, 1, 3, 5, 2]))
        self._env_colors = dict(zip(range(self.env.matrix.shape[-1]), ['red','black','grey','grey','gold']))

        def _create_circle(self, x, y, r, **kwargs):
            return(self.create_oval(x - r, y - r, x + r, y + r, **kwargs))
        def _create_triangle(self, x, y, m, **kwargs):
            return(self.create_polygon(x, y - m / 2, x + m / 2, y, x - m / 2, y, **kwargs))
        tk.Canvas.create_circle = _create_circle
        tk.Canvas.create_triangle = _create_triangle


        self.mainframe = tk.Frame(master=self.root)
        self.sideframe = tk.Frame(master=self.root)
        self.mainframe.grid(row=0, column=0, padx=20, pady=20)
        self.sideframe.grid(row=0, column=1, padx=20, pady=20)

        Fig = Figure(figsize=(5, 4), dpi=80)
        self.FigSubPlot = Fig.add_subplot(111, xlabel='N step', ylabel='Reward rate', xlim=[1,101], ylim = [-10,30])
        self.figcanvas = FigureCanvasTkAgg(Fig, master=self.sideframe)
        self.iterbox = tk.Spinbox(self.sideframe, width=10,
                                  values = list(range(100,1001,100))+list(range(2000,10001,1000)),
                                  state = 'readonly')
        self.learnbox = tk.Spinbox(self.sideframe, width=10, from_=0.1, to=1, increment=0.1,
                                   textvariable = self.learning_rate, state='readonly')
        self.discbox = tk.Spinbox(self.sideframe, width=10, from_=0.1, to=1, increment=0.1,
                                  textvariable = self.discount_factor, state='readonly')
        self.randbox = tk.Spinbox(self.sideframe, width=10, values=list(range(1,11))+list(range(20,101,5)),
                                 textvariable = self.random_factor, state='readonly')
        self.pausebox = tk.Checkbutton(self.sideframe, text="Pause on event", variable=self.pause_on_evet)
        self.startbtn = ttk.Button(master=self.sideframe, text='Start (F5)', command=self.start_run)
        self.stopbtn = ttk.Button(master=self.sideframe, text='Stop (F6)', command=self.stop, state='disabled')
        self.exitbtn = ttk.Button(master=self.sideframe, text='Exit', command=self.exit)
        self.resetbtn = ttk.Button(master=self.sideframe, text='Reset world (F7)', command = self.reset_world,
                                   state='disabled' if self.env_default else 'normal')
        self.histbtn = ttk.Button(master=self.sideframe, text='Show history (F8)', command = self.show_plot_history,
                                  state='disabled')

        self.figcanvas.get_tk_widget().grid(row=0, column=0, columnspan=2)
        ttk.Label(self.sideframe, text="Wins: ", padding=10).grid(row=1, column=0, sticky='e')
        ttk.Label(self.sideframe, textvariable=self.wins, padding=10).grid(row=1, column=1, sticky='w')
        ttk.Label(self.sideframe, text="Deaths: ", padding=10).grid(row=2, column=0, sticky='e')
        ttk.Label(self.sideframe, textvariable=self.deaths, padding=10).grid(row=2, column=1, sticky='w')
        ttk.Label(self.sideframe, text="Total reward: ", padding=10).grid(row=3, column=0, sticky='e')
        ttk.Label(self.sideframe, textvariable=self.sumreward, padding=10).grid(row=3, column=1, sticky='w')
        ttk.Label(self.sideframe, text="Learning Rate: ", padding=10).grid(row=4, column=0, sticky='e')
        self.learnbox.grid(row=4, column=1, sticky='w')
        ttk.Label(self.sideframe, text="Discount Factor: ", padding=10).grid(row=5, column=0, sticky='e')
        self.discbox.grid(row=5, column=1, sticky='w')
        ttk.Label(self.sideframe, text="Random Factor (%): ", padding=10).grid(row=6, column=0, sticky='e')
        self.randbox.grid(row=6, column=1, sticky='w')
        ttk.Label(self.sideframe, text="Number of iterations: ", justify='right', padding=10).grid(row=7, column = 0, sticky='e')
        self.iterbox.grid(row=7, column=1, padx=10, sticky='w')
        self.startbtn.grid(row=8,column=0, padx=10, pady=5)
        self.resetbtn.grid(row=8, column=1, padx=10, pady=5)
        self.stopbtn.grid(row=9, column=0, padx=10, pady=5)
        self.histbtn.grid(row=9, column=1, padx=10, pady=5)
        self.pausebox.grid(row=10, column=0, padx=10, pady=5)
        self.exitbtn.grid(row=10, column=1, padx=10, pady=5)
        self.root.bind('<F5>',self.start_run)
        if not self.env_default:
            self.root.bind('<F7>',self.reset_world)

        self.world = tk.Canvas(self.mainframe, width=300, height=300*int(self.env.rownum/self.env.colnum))
        self.messagebox = tk.Text(self.mainframe, height=1, width=20, state='disabled', font=('Helvetica', '14'))
        self.q_move = tk.Canvas(self.mainframe, width=250, height=250*int(self.env.rownum/self.env.colnum))
        self.q_shoot = tk.Canvas(self.mainframe, width=250, height=250*int(self.env.rownum/self.env.colnum))

        self.world.grid(row=1, column=0, columnspan=2)
        ttk.Label(self.mainframe, text="Message: ", justify='right', padding=10, font=('Helvetica', '14')).grid(row=2, column = 0, sticky='e')
        self.messagebox.grid(row=2, column=1, padx=10, sticky='w')
        ttk.Label(self.mainframe, text="Q value for movements", justify='center', padding=10).grid(row=3,column=0)
        ttk.Label(self.mainframe, text="Q value for shoots", justify='center', padding=10).grid(row=3,column=1)
        self.q_move.grid(row=4, column=0, padx=(0, 5))
        self.q_shoot.grid(row=4, column=1, padx=(5, 0))
        for canvas in [self.world,self.q_move,self.q_shoot]:
            w,h = int(canvas['width']), int(canvas['height'])
            for i in range(self.env.colnum):
                for j in range(self.env.rownum):
                    canvas.create_rectangle(w*j/self.env.rownum, h*i/self.env.colnum,
                                            w*j/self.env.rownum + w/self.env.rownum,
                                            h*i/self.env.colnum + h/self.env.colnum, fill='white')

        arrow_x0, arrow_y0 = self.coord_on_canvas(self.world, 0, 0, 1)
        self.agent_circle = self.draw_agent()
        self.arrow_triangle = self.world.create_triangle(arrow_x0,arrow_y0,15,fill='red')
        self.world_objs = self.draw_world()
        self.q_m_objs, self.q_s_objs = self.draw_qvalues()

    def coord_on_canvas(self, canvas, x, y, gridpos=4):
        grid_dic = dict(zip(range(9),[(i / 4, j / 4) for i, j in zip([1, 2, 3] * 3, [1] * 3 + [2] * 3 + [3] * 3)]))
        xg,yg = grid_dic[gridpos]
        w, h = int(canvas['width']), int(canvas['height'])
        x0, y0 = w*x/self.env.rownum*(1-xg) + w*(x+1)/self.env.rownum*(xg),\
                 h - (h*y/self.env.colnum*(yg) + h*(y+1)/self.env.colnum*(1-yg))
        return(x0, y0)

    def to_coords(self,array_tuple):
        return(np.array([a.tolist() for a in array_tuple]).T)

    def draw_world(self):
        world_objs = []
        fontsize = str(int(20/np.max(self.env.matrix.shape[:2]))+6)
        for i in range(self.env.matrix.shape[-1]):
            coord_pairs = self.to_coords(np.where(self.env.matrix[:, :, i] == 1))
            for x, y in coord_pairs:
                x0, y0 = self.coord_on_canvas(self.world, x, y, self._env_grid[i])
                world_objs.append(self.world.create_text(x0,y0,text=self._env_text[i],fill=self._env_colors[i],
                                                         font=('Helvetica', fontsize)))
        return(world_objs)

    def draw_agent(self):
        x,y = self.agent.state[:2]
        x0, y0 = self.coord_on_canvas(self.world, x, y)
        size = 40/max(self.env.matrix.shape[:2])
        return(self.world.create_circle(x0,y0,size,fill='green'))

    def draw_arrow(self):
        x,y = self.agent.arrowpos
        x0, y0 = self.coord_on_canvas(self.world, x, y, 1)
        size = 15
        return(self.world.create_triangle(x0,y0,size,fill='red'))

    def draw_qvalues(self):
        fontsize = str(int(20 / np.max(self.env.matrix.shape[:2])) + 3)
        grid_dic = dict(zip([3, 0, 1, 2, 7, 4, 5, 6], [3,7,5,1] * 2))
        q_m_objs,q_s_objs = [], []
        for i in range(self.agent.qtable.shape[-1]):
            qtable = self.agent.qtable[:,:,i]
            target_canvas = self.q_move if i < 4 else self.q_shoot
            disabled_moves = {'x0':[3,7], 'xmax':[1,5], 'y0':[0,4], 'ymax':[2,6]}
            for x in range(qtable.shape[0]):
                for y in range(qtable.shape[1]):
                    if not ((x==0 and i in disabled_moves['x0']) or
                            (x==self.env.matrix.shape[0]-1 and i in disabled_moves['xmax']) or
                            (y==0 and i in disabled_moves['y0']) or
                            (y == self.env.matrix.shape[1]-1 and i in disabled_moves['ymax'])):
                        q = qtable[x,y] if abs(qtable[x,y])>=0.01 else 0
                        # cplus = str(int(abs(q)*4)) if int(q*4) else ''
                        # c = 'green'+cplus if q > 0 else 'red'+cplus if q < 0 else 'black'
                        c = 'green' if q > 0 else 'red' if q < 0 else 'black'
                        text = '%.2f'%q if q<100 else '%.1f'%q if q < 999 else '%.0f'%q
                        x0, y0 = self.coord_on_canvas(target_canvas, x, y, grid_dic[i])
                        if i < 4:
                            q_m_objs.append(target_canvas.create_text(x0,y0, text=text, fill=c, font=('Helvetica', fontsize)))
                        else:
                            q_s_objs.append(target_canvas.create_text(x0, y0, text=text, fill=c, font=('Helvetica', fontsize)))
        return(q_m_objs,q_s_objs)

    def clear_world(self):
        self.world.delete(self.agent_circle)
        self.world.delete(self.arrow_triangle)
        for wo in self.world_objs:
            self.world.delete(wo)

    def clear_qtables(self):
        for qmo in self.q_m_objs:
            self.q_move.delete(qmo)
        for qso in self.q_s_objs:
            self.q_shoot.delete(qso)

    def show_plot_history(self, event=None):
        self.root.option_add('*Dialog.msg.font', 'Helvetica 9')
        self.tkMessageBox.showinfo(parent=self.root, title='Plot history',
                                   message=('\n').join(self.plot_history))
        self.root.option_clear()

    def reset_world(self, event=None):
        self.clear_world()
        self.env = Environment(self.rownum,self.colnum)
        self.agent = Agent(self.env)
        self.world_objs = self.draw_world()
        self.agent_circle = self.draw_agent()
        self.arrow_triangle = self.draw_arrow()
        self.reset_plot()

    def reset_plot(self):
        self.run_no, self.lines, self.plot_history = 0, [], []
        self.figcanvas.figure.axes[0].cla()
        self.figcanvas.figure.axes[0].set_ylim([-10, 30])
        self.figcanvas.figure.axes[0].set_xlim(1, int(self.iterbox.get())+0)
        self.figcanvas.figure.axes[0].set_xlabel('N step')
        self.figcanvas.figure.axes[0].set_ylabel('Reward rate')
        self.figcanvas.draw()
        self.histbtn.configure(state='disabled')

    def start_run(self, event=None):
        for key in ['<F5>','<F7>']:
            self.root.unbind(key)
        for fun, key in zip([self.stop,self.show_plot_history],['<F6>','<F8>']):
            self.root.bind(key, fun)
        self.root.bind('<F6>', self.stop)
        if self.run_no >= len(self._colors):
            self.reset_plot()
        self.run_no += 1
        self.figcanvas.figure.axes[0].set_xlim(1, int(self.iterbox.get())+1)
        self.lines = [] if self.run_no == 0 else self.lines
        line, = self.FigSubPlot.plot([], [], self._colors[self.run_no-1]+'-')
        self.lines.append(line)
        self.FigSubPlot.legend(["Run %s"%i for i in range(1,self.run_no+1)])
        self.plot_history.append("Run %s - LR: %.1f, DF: %.1f, RF: %.0f"%(self.run_no,
                                                                        float(self.learnbox.get()),
                                                                        float(self.discbox.get()),
                                                                        float(self.randbox.get())))

        for obj in [self.startbtn, self.learnbox, self.discbox, self.randbox,
                    self.iterbox, self.resetbtn, self.histbtn, self.exitbtn]:
            obj.configure(state='disabled')
        self.stopbtn.configure(state='enabled')

        self.stopmode = False
        self.agent.set_learning_rate(float(self.learning_rate.get()))
        self.agent.set_discount_factor(float(self.discount_factor.get()))
        self.agent.set_random_factor(float(self.random_factor.get())/100)
        self.clear_world()
        self.clear_qtables()
        self.q_m_objs, self.q_s_objs = self.draw_qvalues()

        arrow_x0, arrow_y0 = self.coord_on_canvas(self.world, 0, 0, 1)
        self.agent_circle = self.draw_agent()
        self.arrow_triangle = self.world.create_triangle(arrow_x0,arrow_y0,15,fill='red')
        self.world_objs = self.draw_world()

        self.world.update()
        self.world.after(800)
        cumrewards = []
        for i in range(int(self.iterbox.get())):
            if self.stopmode:
                break
            self.world.after(10)
            a = self.agent.get_best_action()
            result = self.agent.take_action(a)

            self.clear_world()
            self.clear_qtables()
            self.world_objs = self.draw_world()
            self.q_m_objs, self.q_s_objs = self.draw_qvalues()
            self.agent_circle = self.draw_agent()
            if self.agent.arrowpos is not None:
                self.arrow_triangle = self.draw_arrow()
            self.world.update()
            self.messagebox.config(state='normal')
            self.messagebox.delete('1.0', 'end')
            self.messagebox.insert('1.0',result if result != 'nope' else '')
            self.messagebox.config(state='disabled')
            self.messagebox.update()
            if result != 'nope' and self.pause_on_evet.get():
                self.mainframe.after(490)
            for stringvar, value in zip([self.wins, self.deaths, self.sumreward],
                                        [self.agent.wins, self.agent.deaths,np.sum(self.agent.history)]):
                stringvar.set(str(value))
            cumrewards.append(np.sum(self.agent.history)/(i+1))
            if not self.stopmode:
                self.lines[self.run_no - 1].set_data(range(1, i + 2), cumrewards)
                self.figcanvas.draw()
        if not self.stopmode:
            self.stop()
            self.tkMessageBox.showinfo(parent=self.root, title='Info', message='Execution ended')
        else:
            self.tkMessageBox.showinfo(parent=self.root, title='Info', message='Execution interrupted')


    def stop(self, event=None):
        self.stopmode = True
        for key in ['<F6>','<F8>']:
            self.root.unbind(key)
        self.agent.reset_state()
        self.agent.reset_knowlegde()
        self.clear_world()
        self.world.delete(self.agent_circle)
        self.world.delete(self.arrow_triangle)
        arrow_x0, arrow_y0 = self.coord_on_canvas(self.world, 0, 0, 1)
        self.agent_circle = self.draw_agent()
        self.arrow_triangle = self.world.create_triangle(arrow_x0, arrow_y0, 15, fill='red')
        self.world_objs = self.draw_world()
        self.messagebox.config(state='normal')
        self.messagebox.delete('0.1', 'end')
        self.messagebox.config(state='disabled')
        for obj in [self.startbtn, self.learnbox, self.discbox, self.randbox,
                    self.iterbox, self.histbtn, self.exitbtn]:
            obj.configure(state='readonly')
        if not self.env_default:
            self.resetbtn.configure(state='readonly')
        self.stopbtn.configure(state='disabled')
        self.root.bind('<F5>',self.start_run)
        if not self.env_default:
            self.root.bind('<F7>',self.reset_world)

    def exit(self, event=None):
        if self.tkMessageBox.askokcancel(parent=self.root, title='Quit', message='Do you really want to quit?'):
            self.stop()
            self.root.destroy()

    def main_loop(self):
        # Center the window and set the minimal size
        self.root.update()
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        size = tuple(int(numb)+30 for numb in self.root.geometry().split('+')[0].split('x'))
        x = w/2 - size[0]/2
        y = h/2 - size[1]/2
        self.root.geometry("%dx%d+%d+%d" % (size + (x, y)))
        self.root.update()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())
        self.root.mainloop()

class ParamDlg():
    def __init__(self):
        if pyver > 2:
            import tkinter as tk
            import tkinter.ttk as ttk
            import tkinter.messagebox as tkMessageBox
        else:
            import Tkinter as tk
            import tkMessageBox
            import ttk

        self.tk = tk
        self.ttk = ttk
        self.tkMessageBox = tkMessageBox

        self.paramdlg = self.tk.Tk()
        self.paramdlg.title('Starting paramteres')
        self.paramdlg.wm_resizable(0,0)

        reward_labeldic = {'win': 'Win', 'wumpus_killed': 'Wumpus killed', 'death': 'Death',
                           'move_nothing': 'Move to an empty cell', 'shoot_nothing': 'Shoot to an empty cell'}
        self.world_paramdic = {"Classic Wumpus World": True, "Random World": False}
        self.reward_values = {'win':100, 'wumpus_killed':10, 'death':-10,
                           'move_nothing':-1, 'shoot_nothing':-1}

        self.reward_vars = dict([(r,self.tk.StringVar()) for r in reward_labeldic.keys()])
        for r in self.reward_vars.keys():
            self.reward_vars[r].set(str(self.reward_values[r]))
        self.world_param = self.tk.StringVar()
        self.OK = False

        self.ttk.Label(master=self.paramdlg,text='World:').grid(row=0,column=0,sticky='e')
        cbox = self.ttk.Combobox(master=self.paramdlg, values=list(self.world_paramdic.keys()),
                          textvariable=self.world_param,state='readonly')
        cbox.current(1)
        cbox.grid(row=0,column=1,padx=5,pady=5)
        self.ttk.Label(master=self.paramdlg, text='Rewards', font=('Helvetica,16')).grid(row=1, column=0, columnspan=2)
        for i,r in enumerate(self.reward_vars.keys()):
            self.ttk.Label(master=self.paramdlg,text=reward_labeldic[r]+':').grid(row=i+2,column=0, sticky='e', padx=5)
            self.tk.Spinbox(master=self.paramdlg, from_=-100, to=100, width=10,
                            textvariable=self.reward_vars[r]).grid(row=i+2,column=1, sticky='w', padx=2, pady=2)
        self.ttk.Button(master=self.paramdlg, text='OK', command=self.get_values).grid(row=i+3,column=0)
        self.ttk.Button(master=self.paramdlg, text='Cancel', command=self.paramdlg.destroy).grid(row=i + 3, column=1)
        self.paramdlg.bind('<Return>',self.get_values)

        self.paramdlg.mainloop()

    def get_values(self, event=None):
        error = False
        for r in self.reward_values.keys():
            try:
                self.reward_values[r] = int(self.reward_vars[r].get())
            except:
                error = True
        if error:
            self.tkMessageBox.showerror(master=self.paramdlg, title='Error!', message='Rewards should be integers!')
        else:
            self.world_param = self.world_paramdic[self.world_param.get()]
            self.paramdlg.destroy()
            self.OK = True


if __name__ == '__main__':
    params =  ParamDlg()
    if params.OK:
        gui = GUI(env_default=params.world_param, reward_params=params.reward_values)
        gui.main_loop()
