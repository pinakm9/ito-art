import cv2
import numpy as np
from numpy.core.fromnumeric import partition
import tables
import matplotlib.pyplot as plt
import shutil, os, copy
import utility as ut

class Ensemble:
    """
    Description:
        Ensemble of particles used to create the animation

    Attributes:
        img_path: path to the image containing the final frame
        record_path: path to store generated motion data
        particles: particles for creating the animation
        sde: a stochastic differential equation describing the dynamics of the particles

    Methods:
        rotate: rotates the particles by the given angle
        get_dark_pixels: extracts real-valued coordinates of the darker pixels in the final frame of the animation
        move: moves the particles according to the given SDE
    """
    def __init__(self, img_path, record_path, sde):
        self.img_path = img_path
        self.img_name = os.path.basename(img_path).split('.')[0]
        self.record_path = record_path
        self.sde = sde
        self.motion_path = self.record_path + '/{}_to_{}.h5'.format(self.img_name, self.sde.name)
        if not os.path.isdir(record_path):
            os.mkdir(record_path)
        self.color_palette = np.array(['#ff6f69', '#00b159', '#eb6841', '#0e9aa7', '#ffea04'])

    def rotate(self, angle):
        """
        Description:
            rotates the particles by the given angle

        Args:
            angle: angle in radians to rotate by
        """
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.particles = np.array([R @ particle for particle in self.particles])

    def get_dark_pixels(self, threshold, max_particles=2000):
        """
        Description:
            extracts real-valued coordinates of the darker pixels in the final frame of the animation

        Args:
            threshold: integer threshold to determine if a pixel is dark
            max_particles: maximum number of pixels to select
        """
        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.particles = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] < threshold:
                    self.particles.append([i/img.shape[0], j/img.shape[1]])
        self.rotate(3.0 * np.pi / 2.0)
        if max_particles > len(self.particles):
            max_particles = len(self.particles)
        idx = np.random.choice(len(self.particles), size=max_particles, replace=False)
        self.particles = self.particles[idx]
        self.set_colors()
        

    def move(self, specs):
        """
        Description:
            moves the particles according to the given SDE and saves the motion data in a .h5 file

        Args:
            specs: list of 3-tuples specifying how to move forward and save, (num_frames, time_step, save_gap)
        """
        self.frames = self.sde.evolve(self.particles, self.motion_path, specs)


    def set_colors(self, mode='radial'):
        """
        Description:
            assigns colors to particles from the color palette

        Args:
            mode: 'vertical' or 'horizontal' or 'radial' deciding color assignment on the last frame
        """
        d = (1.0 - (-1.0)) / len(self.color_palette)
        self.colors = np.zeros(len(self.particles), dtype=np.int32)
        if mode == 'radial':
            particles = copy.deepcopy(self.particles)
            for i, c in enumerate(self.color_palette):
                box = (i+1) * np.array([[-d/2., d/2.], [-d/2., d/2.]])
                idx = self.in_box(box, particles)
                if len(idx) > 0:
                    particles[idx] = [[np.nan, np.nan]] * len(idx)
                    self.colors[idx] = [i] * len(idx)
        self.colors = self.color_palette[self.colors]
    

    def in_box(self, box, particles):
        """
        Description:
            find out which particles are inside the given box

        Args:
            box: the bounding box as 2x2 matrix
            particles: particles to check containment for

        Returns:
            indices of the contained particles
        """
        idx = []
        for i, p in enumerate(particles):
            if box[0][0] <= p[0] <= box[0][1] and box[1][0] <= p[1] <= box[1][1]:
                idx.append(i)
        return idx
    
    @ut.timer
    def animate(self, fps=24):
        """
        Description:
            uses backward motion to create animation and saves it as a gif
        
        Args:
            num_frames: number of frames to keep in the animation
        """
        motion = tables.open_file(self.motion_path, 'r')
        _, self.time_step, self.ensemble_size, self.dim = motion.root.config.read()[0]
        #frames = list(range(0, self.final_frame, 1))#int(self.final_time / num_frames)))
        frames_folder = self.record_path + '/frames_folder'
        if not os.path.isdir(frames_folder):
            os.mkdir(frames_folder)
        
        fig = plt.figure(figsize=(8, 8), frameon=False)
        ax = fig.add_subplot(111) 
        fig.patch.set_visible(False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        #ax.set_aspect(1)
        def update_plot(frame):
            ax.clear()
            # read data to plot
            ensemble = getattr(motion.root.ensemble, 'time_' + str(frame)).read()
            if self.dim == 2:
                ax.scatter(ensemble[:, 0], ensemble[:, 1], c=self.colors)
            else:
                ax.scatter(ensemble[:, 0], ensemble[:, 1], ensemble[:, 2], c=self.colors, s=3.0)
            """
            ax.set_title('time = {:.2f}'.format(frame * self.time_step))
            ax.set_xlabel('x')
            if ax_lims[0] is not None:
                ax.set_xlim(ax_lims[0])
            ax.set_ylabel('y')
            if ax_lims[1] is not None:
                ax.set_xlim(ax_lims[1])
            if self.dim == 3:
                ax.set_zlabel('z')
                if ax_lims[2] is not None:
                    ax.set_xlim(ax_lims[2])
            """
            plt.tight_layout()
            plt.savefig(frames_folder + '/frame_{}.png'.format(frame))
            print('Frame {} has been drawn.'.format(frame), end='\r')
            
        for frame in self.frames:
            update_plot(frame)

        height, width, _ = cv2.imread(frames_folder + '/frame_0.png').shape
        video_path = self.record_path + '/{}_to_{}'.format(self.img_name, self.sde.name) + '.mp4'
        video = cv2.VideoWriter(video_path, fourcc = cv2.VideoWriter_fourcc(*'mp4v'), frameSize=(width,height), fps=fps)
        for frame in self.frames[::-1]:
            video.write(cv2.imread(frames_folder + '/frame_{}.png'.format(frame)))
        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(frames_folder)
        motion.close()


        