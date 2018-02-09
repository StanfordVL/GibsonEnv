import pygame
from pygame import surfarray
from pygame.surfarray import pixels3d
import time
import numpy as np
import cv2
import os
from datetime import datetime
from PIL import Image
import scipy.misc
from gibson.core.render.profiler import Profiler
from enum import Enum

class View(Enum):
    EMPTY = 0
    RGB_FILLED = 1
    RGB_PREFILLED = 2
    DEPTH = 3
    NORMAL = 4
    SEMANTICS = 5
    PHYSICS = 6


class SimpleUI():
    '''Static UI'''
    def __init__(self, width_col, height_col, windowsz, env = None):
        self.env = env
        self.width  = width_col * windowsz
        self.height = height_col * windowsz
        self.windowsz = windowsz
        self.screen = pygame.display.set_mode([self.width, self.height], 0, 32)
        self.screen_arr = np.zeros([self.width, self.height, 3])
        self.screen_arr.fill(255)
        self.is_recording = False
        self.components = [View[item] for item in self.env.config["ui_components"]]
        self._add_all_images()
        self.record_root = None


    def _add_all_images(self):
        for index, component in enumerate(self.components):
            img = np.zeros((self.windowsz, self.windowsz, 3))
            img.fill(np.random.randint(0, 256))
            self._add_image(img, self.POS[index][0], self.POS[index][1])

    def update_view(self, view, tag):
        assert(tag in self.components), "Invalid view tag " + view
        for index, component in enumerate(self.components):
            if tag == component:
                self._add_image(
                    np.swapaxes(view, 0, 1), 
                    self.POS[index][0],
                    self.POS[index][1])
                return

    def _add_image(self, img, x, y):
        #self.screen.blit(img, (x, y))
        self.screen_arr[x: x + img.shape[0], y:y + img.shape[1], :] = img

    def clear(self):
        self.screen_arr.fill(255)
        self.refresh()

    def refresh(self):
        if "enable_ui_recording" in self.env.config:
            cmd=cv2.waitKey(5)%256
            if cmd == ord('r'):
                self.start_record()
            if cmd == ord('q'):
                self.end_record()

            img = np.uint8(self.screen_arr)
            cv2.imshow("Recording", img)
            if self.is_recording:
                self.curr_output.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #with Profiler("Refreshing"):
        pygame.display.flip()
        surfarray.blit_array(self.screen, self.screen_arr)
        #surf = pygame.surfarray.make_surface(self.screen_arr)
        #self.screen.blit(surf, (0, 0))
        #pygame.display.update()


    def start_record(self):
        print("start recording")
        if self.is_recording:
            return    # prevent double enter
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # 'XVID' smaller
        file_keyword = datetime.now()
        filename = 'record-{}.avi'.format(file_keyword)
        filepath = os.path.join(self.RECORD_ROOT, filename)

        foldername = 'record-{}'.format(file_keyword)
        folderpath = os.path.join(self.RECORD_ROOT, foldername)

        #os.mkdir(folderpath)
        self.curr_output = cv2.VideoWriter(filepath, fourcc, 22.0, self.UI_DIM)

        self.is_recording = True

    def make_video(self):
        return

    def end_record(self):
        print("end recording")
        self.curr_output.release()
        self.is_recording = False
        return

class OneViewUI(SimpleUI):
    '''UI with four modalities, default resolution
    One: Center,
    '''
    def __init__(self, windowsz=256, env = None):
        self.POS = [
            (0, 0)                 # One
        ]
        SimpleUI.__init__(self, 1, 1, windowsz, env)

class TwoViewUI(SimpleUI):
    '''UI with four modalities, default resolution
    One: Left,
    Two: Right
    '''
    def __init__(self, windowsz=256, env = None):
        self.POS = [
            (0, 0),                 # One
            (windowsz, 0)           # Two
        ]
        SimpleUI.__init__(self, 2, 1, windowsz, env)

class ThreeViewUI(SimpleUI):
    '''UI with four modalities, default resolution
    One:    left
    Two:    center
    Three:  right
    '''
    def __init__(self, windowsz=256, env = None):
        self.POS = [
            (0, 0),                 # One
            (windowsz, 0),          # Two
            (windowsz * 2, 0)       # Three
        ]
        SimpleUI.__init__(self, 3, 1, windowsz, env)

class FourViewUI(SimpleUI):
    '''UI with four modalities, default resolution
    One:    top left
    Two:    top right
    Three:  bottom left
    Four:   bottom right
    '''
    def __init__(self, windowsz=256, env = None):
        self.POS = [
            (0, 0),                 # One
            (0, windowsz),          # Two
            (windowsz, 0),          # Three
            (windowsz, windowsz)    # Four
        ]
        SimpleUI.__init__(self, 2, 2, windowsz, env)


def main6():
    UI = SimpleUI(768, 768)

    ## Center left top
    grey_1 = np.zeros((512, 512, 3))
    grey_1.fill(100)

    ## Right top
    grey_2 = np.zeros((256, 256, 3))
    grey_2.fill(120)

    ## Right mid
    grey_3 = np.zeros((256, 256, 3))
    grey_3.fill(140)

    ## Bottom left
    grey_4 = np.zeros((256, 256, 3))
    grey_4.fill(120)

    ## Bottom mid
    grey_5 = np.zeros((256, 256, 3))
    grey_5.fill(180)

    ## Bottom right
    grey_6 = np.zeros((256, 256, 3))
    grey_6.fill(200)


    UI = SixViewUI()
    rgb = np.zeros((512, 512, 3))
    rgb.fill(0)

    while True:
        UI.refresh()
        UI.update_rgb(rgb)
        rgb += 20
        time.sleep(0.2)
        #screen_arr = 255 - screen_arr


def main4():
    ## Center left top
    windowsz = 512
    grey_1 = np.zeros((windowsz, windowsz, 3))
    grey_1.fill(100)

    ## Right top
    grey_2 = np.zeros((windowsz, windowsz, 3))
    grey_2.fill(120)

    ## Right mid
    grey_3 = np.zeros((windowsz, windowsz, 3))
    grey_3.fill(140)

    ## Bottom left
    grey_4 = np.zeros((windowsz, windowsz, 3))
    grey_4.fill(160)

    UI = FourViewUI(windowsz)
    rgb = np.zeros((windowsz, windowsz, 3))
    rgb.fill(0)

    UI.update_view(grey_1, View.RGB_FILLED)
    UI.update_view(grey_2, View.DEPTH)
    UI.update_view(grey_3, View.NORMAL)
    UI.update_view(grey_4, View.SEMANTICS)

    while True:
        UI.refresh()
        UI.update_view(rgb, View.RGB_FILLED)
        rgb += 20
        time.sleep(0.2)
        #screen_arr = 255 - screen_arr

def main2():
    ## Center left top
    grey_1 = np.zeros((256, 256, 3))
    grey_1.fill(100)

    ## Right top
    grey_2 = np.zeros((256, 256, 3))
    grey_2.fill(120)

    UI = TwoViewUI()
    rgb = np.zeros((256, 256, 3))
    rgb.fill(0)

    UI.update_physics(grey_1)
    UI.update_rgb(grey_2)

    while True:
        UI.refresh()
        UI.update_rgb(rgb)
        rgb += 20
        time.sleep(0.2)
        #screen_arr = 255 - screen_arr

if __name__ == "__main__":
    #main6()
    #main2()
    main4()


"""
##### Deprecated
class SixViewUI(SimpleUI):
    '''UI with all modalities, default resolution
    RGB:       512x512, (top left)
    Map:       256x256, (top right)
    Physics:   256x256, (center right)
    Depth:     256x256, (bottom left)
    Semantics: 256x256, (bottom right)
    Normal:    256x256  (bottom right)
    '''
    UI_DIM    = (768, 768)
    POS_RGB   = (0, 0)
    POS_PHYSICS = (512, 256)
    POS_MAP   = (512, 0)
    POS_DEPTH = (0, 512)
    POS_SEM   = (512, 512)
    POS_SURF  = (256, 512)
    def __init__(self):
        SimpleUI.__init__(self, 768, 768)
        self._add_all_images()


    def add_all_images(self):
        img_rgb = np.zeros((512, 512, 3))
        img_map = np.zeros((256, 256, 3))
        img_physics = np.zeros((256, 256, 3))
        img_sem = np.zeros((256, 256, 3))
        img_surf = np.zeros((256, 256, 3))
        img_depth = np.zeros((256, 256, 3))

        img_rgb.fill(100)
        img_map.fill(120)
        img_physics.fill(140)
        img_depth.fill(120)
        img_sem.fill(180)
        img_surf.fill(200)

        self.add_image(img_rgb, self.POS_RGB[0], self.POS_RGB[1])
        self.add_image(img_sem, self.POS_SEM[0], self.POS_SEM[1])
        self.add_image(img_depth, self.POS_DEPTH[0], self.POS_DEPTH[1])
        self.add_image(img_physics, self.POS_PHYSICS[0], self.POS_PHYSICS[1])
        self.add_image(img_surf, self.POS_SURF[0], self.POS_SURF[1])
        self.add_image(img_map, self.POS_MAP[0], self.POS_MAP[1])

    def update_rgb(self, rgb):
        #rgb = pygame.transform.rotate(rgb, 90)
        self.add_image(np.swapaxes(rgb, 0, 1), self.POS_RGB[0], self.POS_RGB[1])

    def update_sem(self, sem):
        #sem = pygame.transform.rotate(sem, 90)
        self.add_image(np.swapaxes(sem, 0, 1), self.POS_SEM[0], self.POS_SEM[1])

    def update_physics(self, physics):
        #physics = pygame.transform.rotate(physics, 90)
        self.add_image(np.swapaxes(physics, 0, 1), self.POS_PHYSICS[0], self.POS_PHYSICS[1])

    def update_depth(self, depth):
        #depth = pygame.transform.rotate(depth, 90)
        self.add_image(np.swapaxes(depth, 0, 1), self.POS_DEPTH[0], self.POS_DEPTH[1])

    def update_normal(self, surf):
        self.add_image(np.swapaxes(surf, 0, 1), self.POS_SURF[0], self.POS_SURF[1])

    def update_map(self, map_img):
        self.add_image(np.swapaxes(map_img, 0, 1), self.POS_MAP[0], self.POS_MAP[1])
"""