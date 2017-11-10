import pygame
from pygame import surfarray
from pygame.surfarray import pixels3d
import time
import numpy as np

class SimpleUI():
    '''Static UI'''
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.screen = pygame.display.set_mode([width, height], 0, 32)
        self.screen_arr = np.zeros([width, height, 3])
        self.screen_arr.fill(255)

    def add_image(self, img, x, y):
        self.screen_arr[x: x + img.shape[0], y:y + img.shape[1], :] = img

    def clear(self):
        self.screen_arr.fill(255)
        self.refresh()

    def refresh(self):
        pygame.display.flip()
        surfarray.blit_array(self.screen, self.screen_arr)


def main():
    UI = SimpleUI(680, 532)
    #flash = np.zeros((512, 512, 3))
    #flash.fill(255)
    green = np.zeros((512, 512, 3))
    green[:, :, 1] = 255
    
    grey_1 = np.zeros((512, 512, 3))
    grey_1.fill(100)

    grey_2 = np.zeros((128, 128, 3))
    grey_2.fill(100)

    while True:
        #flash = 255 - flash
        UI.add_image(grey_1, 0, 0)
        UI.add_image(grey_2, 532, 0)

        UI.add_image(grey_2, 532, 128 + 40)
        UI.add_image(grey_2, 532, 128 + 40 + 128 + 40)
        
        UI.refresh()
        time.sleep(0.2)
        #screen_arr = 255 - screen_arr

if __name__ == "__main__":
    main()