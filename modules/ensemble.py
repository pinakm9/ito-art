import cv2
import numpy as np
import tables
import matplotlib.pyplot as plt
import shutil, os
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
        idx = np.random.choice(len(self.particles), size=max_particles, replace=False)
        self.particles = self.particles[idx]

    def move(self, begin_specs, end_specs):
        """
        Description:
            moves the particles according to the given SDE and saves the motion data in a .h5 file

        Args:
            begin_specs: 3-tuple specifying how to move forward and save in the beginning (num_frames, time_step, save_gap)
            end_specs: 3-tuple specifying how to move forward and save in the end (num_frames, time_step, save_gap)
        """
        self.frames = self.sde.evolve(self.particles, self.motion_path, begin_specs, end_specs)

    @ut.timer
    def animate(self, num_frames):
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
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111) 
        
        def update_plot(frame):
            ax.clear()
            # read data to plot
            ensemble = getattr(motion.root.ensemble, 'time_' + str(frame)).read()
            if self.dim == 2:
                ax.scatter(ensemble[:, 0], ensemble[:, 1])
            else:
                ax.scatter(ensemble[:, 0], ensemble[:, 1], ensemble[:, 2], c='b', s=3.0)
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
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.savefig(frames_folder + '/frame_{}.png'.format(frame))
            print('Frame {} has been drawn.'.format(frame), end='\r')
            
        for frame in self.frames:
            update_plot(frame)

        height, width, _ = cv2.imread(frames_folder + '/frame_0.png').shape
        video_path = self.record_path + '/{}_to_{}'.format(self.img_name, self.sde.name) + '.mp4'
        video = cv2.VideoWriter(video_path, fourcc = cv2.VideoWriter_fourcc(*'mp4v'), frameSize=(width,height), fps=24)
        for frame in self.frames[::-1]:
            video.write(cv2.imread(frames_folder + '/frame_{}.png'.format(frame)))
        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(frames_folder)
        motion.close()


        