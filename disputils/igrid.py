# import the necessary pacakages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

"""
Modulename: igrid -> facilitaes creation and populating a grid with images
"""

class  Igrid:
    """
    Class : Igrid
        creates a grid of rows x cols to hold images
        Arguments:
            rows : int
            cols : int
            size : (optional) size of matplotlib figure in inches.
                    default is 8 inches
            cmap : (optional) colour map to use.
                    default is "Greys" but reversed (_r)
    """
    def __init__(self,rows,cols,size=8, cmap='Greys_r'):
        """
        Igrid constructor
        """
        self.rows = rows
        self.cols = cols
        self.size = size
        self.fig = plt.figure(figsize=(self.size, self.size), dpi=100)
        self.grid = gridspec.GridSpec(self.rows,self.cols, wspace=0.0, hspace=0.0,
                left=None,right=None, top=None, bottom=None)
        self.cmap = cmap

    # set image to a particular cell
    def set(self,row,col,img,draw=True):
        """
        draws an image at cell location given by row and col
        if draw is False then it will just draw the grid
        which useful while experimenting
        """
        i = row * self.cols + col
        ax = plt.Subplot(self.fig, self.grid[i])
        if draw:
            ax.imshow(img, cmap=self.cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.add_subplot(ax)

    # display plot on screen
    def show(self):
        """
        Renders the figure on screen
        """
        #plt.subplots_adjust(left=None, right=None, top=None, bottom=None)
        plt.show()
    # close plot on screen
    def close(self):
        """
        close the figure
        """
        plt.close('all')

    # save to file
    def save(self,fname="figure.png"):
        """
        Save figure to filename given by fname. Deafault name is "figure.png"
        """
        self.fig.savefig(fname)
