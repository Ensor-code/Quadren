import matplotlib.pyplot as plt
import numpy             as np
import numba
import os
import re

from scipy.spatial.transform import Rotation
from ipywidgets              import interact, FloatSlider
from glob                    import glob
from tqdm                    import tqdm


class Quadren():

    def __init__(self, points, xyz_dim, xyz_min=None, xyz_max=None):
        """
        Create a cartesian box.

        Parameters
        ----------
        points  : n by 3 array 
        xyz_dim : dimensions
        """
        # Set the dims of the scene
        self.xyz_dim = xyz_dim
        
        if (xyz_min is None) and (xyz_max is None):
            # Compute size of the box
            self.xyz_min = np.min(points, axis=0)
            self.xyz_max = np.max(points, axis=0)
        else:
            # Or take the one that was given
            self.xyz_min = xyz_min
            self.xyz_max = xyz_max
            
        # Set a tolerance (which should be smaller than the resolution of the cube)
        tol = 1.0e-4 / self.xyz_dim
        
        # Compute xyz size of the box (1 tol'th larger)
        self.xyz_len = (1.0 + tol) * (self.xyz_max - self.xyz_min)
        
        # Normalize the point coordinates to be in [0,1]
        self.points_normed = (points - self.xyz_min) * (1.0 / self.xyz_len)
        
        # Count the number of points in each octree grid cell
        self.num = Quadren.get_number_array(self.points_normed, self.xyz_dim)


    def map_data(self, data, interpolate=True):
        """
        Map data to the cube.
        """
        # Integrate (i.e. sum) the point data over each cell
        dat = Quadren.integrate_data(self.points_normed, self.xyz_dim, data)
        
        # Divide by the number of points in each cell to get the average
        dat = np.divide(dat, self.num, out=dat, where=(self.num!=0))
        
        # Smooth data cube
        # dat = kernel_1(dat, self.num)
        # dnm = kernel_2(     self.num)
        # dat = np.divide(dat, dnm, out=dat, where=(dnm!=0.0))
        
        # Return data in a cube
        return dat


    @staticmethod
    @numba.njit(parallel=True)
    def get_number_array(points_normed, xyz_dim):
        """
        Compute the number of points that live in each cell.
        """
        # Create number array
        num = np.zeros((xyz_dim[0], xyz_dim[1], xyz_dim[2]), dtype=np.int64)
        
        # Compute the indices of the points in the octree grid
        indices = (points_normed * xyz_dim).astype(np.int64)
        
        # Count the number of points at every index
        for ix, iy, iz in indices:
            num[ix, iy, iz] += 1
        
        # Return the number array
        return num

    
    @staticmethod
    @numba.njit(parallel=True)
    def integrate_data(points_normed, xyz_dim, data):
        """
        Sum the data in each cell.
        """
        # Create dat array
        dat = np.zeros((xyz_dim[0], xyz_dim[1], xyz_dim[2]), dtype=data.dtype)
        
        # Compute the indices of the points in the octree grid
        indices = (points_normed * xyz_dim).astype(np.int64)
        
        # Add the data at every index
        for i, (ix, iy, iz) in enumerate(indices):
            dat[ix, iy, iz] += data[i]
        
        # Return the integrated data
        return dat

    
    def render(self, dat):
        """
        Render an image of the data along the 3rd axis.
        """
        return Renderer.render__(self.xyz_dim, dat)

    
    @staticmethod
    @numba.njit(parallel=True)
    def render__(xyz_dim, dat):
        """
        Render an image of the data along the 3rd axis.
        
        This is the simplest implementation: just sum along 3rd axis.
        
        Parameters
        ----------
        xyz_dim : 1D array, shape=(3,)
            xyz dimensions of the data array.
        dat : 3D array, shape=xyz_dim
            data array that is to be rednered.
            
        Returns
        -------
        img : 2D array, shape
            data collapsed along the 3rd axis.
        """
        # Create number matrices
        img = np.zeros((xyz_dim[0], xyz_dim[1]), dtype=dat.dtype)
        
        # Integrate along the z axis
        for i in range(xyz_dim[0]):
            for j in range(xyz_dim[1]):
                for k in range(xyz_dim[2]):
                    img[i,j] = img[i,j] + dat[i,j,k]
        
        # Return the integrated data
        return img