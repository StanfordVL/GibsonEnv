import time, random
import math
import collections
from collections import deque
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from multiprocessing import Process, Pipe
#import gobject

class SmoothList(collections.MutableSequence):
    max_entries = 200
    def __init__(self):
        self.list = []

    def __getitem__(self, idx):
        scale = 1
        if len(self.list) > SmoothList.max_entries:
            scale = len(self.list) / float(SmoothList.max_entries)
        return self.list[int((idx + 1) * scale) - 1]

    def __len__(self):
        return min(len(self.list), SmoothList.max_entries)

    def append(self, elem):
        '''
        if len(self.list) >= SmoothList.max_entries:
            newlist = []
            i = 0
            while i < len(self.list):
                newlist.append((self.list[i] + self.list[i + 1]) / 2.0)
                i = i + 2
            self.list = newlist
        '''
        self.list.append(elem)

    def __delitem__(self, i): del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)


class RealtimePlot:
    def __init__(self, axes, max_entries = 100, full = False):
        self.axes = axes
        self.max_entries = max_entries
        if not full:
            self.axis_x = deque(maxlen=max_entries)
            self.axis_y = deque(maxlen=max_entries)
            self.lineplot, = axes.plot([], [], "bo-")
        else:
            self.axis_x = SmoothList()
            self.axis_y = SmoothList()
            self.lineplot, = axes.plot([], [], "ro-")
                
        self.axes.set_autoscaley_on(True)

    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)
        self.lineplot.set_data(self.axis_x, self.axis_y)
        self.axes.set_xlim(self.axis_x[0], self.axis_x[-1] + 1e-15)
        self.axes.relim(); self.axes.autoscale_view() # rescale the y-axis

    def animate(self, figure, callback, interval = 50):
        import matplotlib.animation as animation
        def wrapper(frame_index):
            self.add(*callback(frame_index))
            self.axes.relim(); self.axes.autoscale_view() # rescale the y-axis
            return self.lineplot
        animation.FuncAnimation(figure, wrapper, interval=interval)


class RewardDisplayer:
    def __init__(self):
        self.all_rewards = []
        self.factor = 3
        font0 = FontProperties()
        gs = gridspec.GridSpec(12, 3)

        fig = plt.figure(figsize=(10.25, 8))
        
        self.axes_text = plt.subplot(gs[:2, :], facecolor='gainsboro')
        self.axes_full = plt.subplot(gs[2:9, :], facecolor='gainsboro')
        self.axes_real = plt.subplot(gs[9:12, :], facecolor='gainsboro')

        self.axes_text.set_xlim(0, 10)
        self.axes_text.set_ylim(0, 10)

        #ax1.set_title('title1', color='c', rotation='vertical',x=-0.1,y=0.5)
        self.axes_full.legend(loc='upper left', handles=[mpatches.Patch(color='red', label='Score - Time Plot')], prop={'size': 13})
        self.axes_real.legend(loc='upper left', handles=[mpatches.Patch(color='blue', label='Reward - Time Plot')], prop={'size': 13})
 
        self.font = font0.copy()
        self.font.set_family('sans caption')
        self.font.set_weight('semibold')
        self.alignment = {'horizontalalignment': 'left', 'verticalalignment': 'baseline'}
        self.axes_text.axis('off')

        self.display_full = RealtimePlot(self.axes_full, full = True)
        self.display_real = RealtimePlot(self.axes_real)

        self.txt_reward = None
        self.txt_totalr = None
        self.txt_time   = None

        self.axes_text.text(0, 9, 'Reward:', fontproperties=self.font, size = 15,  **self.alignment)
        self.axes_text.text(4.5, 9, 'Score:', fontproperties=self.font, size = 15,  **self.alignment)
        self.axes_text.text(0, 3, 'Time:', fontproperties=self.font, size = 15,  **self.alignment)
        self.score = 0

        self.start = time.time()
            
    def _smooth(self):
        return self.all_rewards

    def reset(self):
        self.score = 0

    def add_reward(self, reward):
        self.score = self.score + reward
        cur_time = time.time() - self.start
        self.display_real.add(cur_time, reward)
        self.display_full.add(cur_time, self.score)
        if self.txt_reward:
            self.txt_reward.remove()
        if self.txt_totalr:
            self.txt_totalr.remove()
        if self.txt_time:
            self.txt_time.remove()
        self.txt_reward = self.axes_text.text(2, 9, '%.2f' % reward, fontproperties=self.font, size = 30,  **self.alignment)
        self.txt_totalr = self.axes_text.text(6.5, 9, '%.2f' % self.score, fontproperties=self.font, size = 30, **self.alignment)
        self.txt_time   = self.axes_text.text(2, 3, '%.2f' % cur_time, fontproperties=self.font, size = 30, **self.alignment)
        plt.pause(0.0001)

    def terminate(self):
        plt.close('all')

    def poll_draw(self):
        while 1:
            #print("polling")
            if not self.pipe.poll():
                break
            command = self.pipe.recv()
            #print("received reward", command)
            if command is None:
                self.terminate()
                return False
            else:
                self.add_reward(command)
            time.sleep(0.01)
        #self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('Starting plotter...')

        self.pipe = pipe
        #self.fig, self.ax = plt.subplots()
        #self.gid = gobject.timeout_add(1000, )
        self.poll_draw()
        print('...done')
        #plt.show()

class MPRewardDisplayer(object):
    def __init__(self):
        self.plot_pipe, plotter_pipe = Pipe()
        self.plotter = RewardDisplayer()
        self.plot_process = Process(target=self.plotter,
                                    args=(plotter_pipe,))
        self.plot_process.daemon = True
        self.plot_process.start()

    def add_reward(self, reward, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send(reward)

    def reset(self):
        return

def main():
    r_displayer = MPRewardDisplayer()
    for i in range(10000):
        num = random.random() * 100 - 30
        r_displayer.add_reward(num)
        if i % 40 == 0:
            r_displayer.reset()

if __name__ == "__main__": main()