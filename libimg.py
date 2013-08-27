import skimage.io as io
import numpy
import re
import math
import copy

Y = 0
X = 1


class NMImage ():
    def __init__(self, file_location=''):
        self.filename = file_location
        self.sizex = 0
        self.sizey = 0
        self.maxval = 255
        self.data = self.open()
        self.size = self.sizey * self.sizex

    def __repr__(self):
        name = self.filename.split("/")[-1]
        return name + " (" + str(self.sizex) + ", " + str(self.sizey) + ")"

    def open(self):
        #http://flockhart.virtualave.net/RBIF0100/regexp.html
        io.use_plugin("freeimage")
        self.data = io.imread(self.filename).astype("uint8")
        self.sizex = self.data.shape[0]
        self.sizey = self.data.shape[1]

        return self.data

    def save(self, name):
        io.use_plugin("freeimage")
        io.imsave(name, self.data)
        # Save ASCII
        '''out = open(name, "w")
        outstring = "P2\n%s %s %s\n" % (self.sizex, self.sizey, "255")
        for y in range(self.sizey):
            for x in range(self.sizex):
                outstring += str(self.data[y, x]) + " "
        out.write(outstring)
        out.close()'''

    def getPixel(self, coord):
        return self.data[coord[Y], coord[X]]

    def putPixel(self, coord, value):
        self.data[coord[Y], coord[X]] = value

    def getCoord(self, position):
        #http://cl.ly/image/380z3r3y1H19
        x = (position % (self.sizex*self.sizey)) % self.sizex
        y = (position % (self.sizex*self.sizey)) / self.sizex
        return (y, x)

    def show(self, title=''):
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        if self.filename.endswith(".pgm"):
            plt.imshow(self.data, cmap=cm.Greys_r)
        else:
            plt.imshow(self.data)
        plt.axis('off')
        if len(title) == 0:
            title = self.filename
        plt.title(title, fontsize=14)
        plt.show()

    def copy(self):
        return copy.deepcopy(self)

    def minmaxvalue(self):
        minval = 99999
        maxval = -99999
        for i in range(self.size):
            pixelvalue = self.getPixel(self.getCoord(i))
            if pixelvalue < minval:
                minval = pixelvalue
            if pixelvalue > maxval:
                maxval = pixelvalue
        return minval, maxval

    def error(self, message, function):
        import sys
        print message, "in", function
        sys.exit(1)

#######################################################
#
#
#                   NOT IO
#
#
#######################################################

    def binarize(self, threshold, min, max):
        for i in range(self.size):
            coord = self.getCoord(i)
            if self.getPixel(coord) > threshold:
                self.putPixel(coord, max)
            else:
                self.putPixel(coord, min)

    def invert(self):
        for i in range(self.size):
            coord = self.getCoord(i)
            value = self.getPixel(coord)
            self.putPixel(coord, self.maxval - value)

    def histogram(self):
        from matplotlib import pyplot as plt
        plt.hist(self.data)
        plt.show()

    def normalize(self, inf, sup):
        output = self.copy()

        minval, maxval = self.minmaxvalue()
        output.maxval = maxval
        output.data.astype('uint8')
        if minval < maxval:
            for p in range(output.size):
                coord = output.getCoord(p)
                numerator = (sup-inf) * (self.getPixel(coord)-minval)
                denominator = (maxval-minval) + inf
                result = numerator / denominator
                result = int(result)
                output.data[coord[Y], coord[X]] = numpy.uint8(result)
        else:
            self.error("Empty image", "normalize")
        return output

    def logTransform(self, c):
        output = self.copy()

        def logfunc(number):
            return math.log(number+1)
        results = numpy.vectorize(logfunc)(output.data)
        output.data = results
        return output

    def expTransform(self):
        pass

    def squareTransform(self):
        pass

    def sqrtTransform(self):
        pass

    def constratTransform(self):
        pass
