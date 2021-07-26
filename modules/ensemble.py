import cv2
import numpy as np


class Ensemble:
    """
    Description:
        Ensemble of particles used to create the animation

    Attributes:
        img_path: path to the image containing the final frame
        data_path: path to store generated motion data
        particles: particles for creating the animation

    Methods:
        rotate: rotates the particles by the given angle
        get_dark_pixels: extracts real-valued coordinates of the darker pixels in the final frame of the animation
        move: moves the particles according to the given SDE
    """
    def __init__(self, img_path, data_path):
        self.img_path = img_path
        self.data_path = data_path

    def rotate(self, angle):
        """
        Description:
            rotates the particles by the given angle

        Args:
            angle: angle in radians to rotate by
        """
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.particles = np.array([R @ particle for particle in self.particles])

    def get_dark_pixels(self, threshold):
        """
        Description:
            extracts real-valued coordinates of the darker pixels in the final frame of the animation

        Args:
            threshold: integer threshold to determine if a pixel is dark
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.particles = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] < threshold:
                    self.particles.append([i/img.shape[0], j/img.shape[1]])
        self.particles =  rotate(self.particles, 3.0 * np.pi / 2.0)

    def move(self, sde):
        """
        Description:
            moves the particles according to the given SDE

        Args:
            sde: a stochastic differential equation describing the dynamics of the particles
        """
        pass
