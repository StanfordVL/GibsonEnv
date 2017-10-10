import math, os, pyglet, sys
from pyglet.gl import *
import time

class World(pyglet.window.Window):
    def __init__(self, scale=10, center_pos=(0, 0, -15), speed=1.0,
                 *args, **kwargs):
        super(World, self).__init__(*args, **kwargs)
        self.scale = scale
        self.center_pos = center_pos
        self.speed = speed
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        self.textures = self.load_textures()
        self.clock = 0
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    @staticmethod
    def load_textures():
        img_dir = 'imgs'
        textures = []
        if not os.path.isdir(img_dir):
            print 'Could not find directory "%s" under "%s"' % (img_dir,
                                                                os.getcwd())
            sys.exit(1)
        for image in os.listdir(img_dir):
            try:
                image = pyglet.image.load(os.path.join(img_dir, image))
            except pyglet.image.codecs.dds.DDSException:
                print '"%s" is not a valid image file' % image
                continue
            #print(image.get_texture())
            textures.append(image.get_texture())

            glEnable(textures[-1].target)
            glBindTexture(textures[-1].target, textures[-1].id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE,
                         image.get_image_data().get_data('RGBA',
                                                         image.width * 4))
        if len(textures) == 0:
            print 'Found no textures to load. Exiting'
            sys.exit(0)
        return textures

    def update(self, _):
        self.on_draw()
        self.clock += .01

    def on_draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        time.sleep(0.1)
        self.draw_images()

    def draw_images(self):
        angle_base = (self.clock * self.speed * 50) % 360
        angle_delta = 360. / len(self.textures)

        for i, texture in enumerate(self.textures):
            angle = math.radians((angle_base + i * angle_delta) % 360)
            dx = math.sin(angle) * self.scale
            dz = math.cos(angle) * self.scale

            if texture.width > texture.height:
                rect_w = texture.width / float(texture.height)
                rect_h = 1
            else:
                rect_w = 1
                rect_h = texture.height / float(texture.width)

            glPushMatrix()
            glTranslatef(dx + self.center_pos[0], self.center_pos[1],
                         dz + self.center_pos[2])
            glBindTexture(texture.target, texture.id)
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 0.0); glVertex3f(-rect_w, -rect_h, 0.0)
            glTexCoord2f(1.0, 0.0); glVertex3f( rect_w, -rect_h, 0.0)
            glTexCoord2f(1.0, 1.0); glVertex3f( rect_w,  rect_h, 0.0)
            glTexCoord2f(0.0, 1.0); glVertex3f(-rect_w,  rect_h, 0.0)
            glEnd()
            glPopMatrix()

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65.0, width / float(height), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)


if __name__ == "__main__":
    window = World(width=800, height=600)
    pyglet.app.run()