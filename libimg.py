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
        self.data = io.imread(self.filename)
        self.sizex = self.data.shape[0]
        self.sizey = self.data.shape[1]

        return self.data

    def save(self, name):
        if self.data.dtype == "float64":
            self.data = self.data.astype("uint8")
        io.use_plugin("freeimage")
        io.imsave(name, self.data)

    def getPixel(self, coord):
        return self.data[coord[Y], coord[X]]

    def putPixel(self, coord, value):
        self.data[coord[Y], coord[X]] = value

    def getCoord(self, position):
        #http://cl.ly/image/380z3r3y1H19
        x = (position % (self.sizex*self.sizey)) % self.sizex
        y = (position % (self.sizex*self.sizey)) / self.sizex
        return (x, y)

    def show(self, title=''):
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        plt.figure()
        if self.filename.endswith("pgm"):
            plt.imshow(self.data, cmap=cm.gray)
        else:
            plt.imshow(self.data)
        plt.axis('off')
        if len(title) == 0:
            title = self.filename
        plt.title(title, fontsize=14)

    def showAll(self):
        from matplotlib import pyplot as plt
        plt.show()

    def histogram(self, title=''):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.hist(self.data)
        if len(title) == 0:
            title = self.filename
        plt.title(title, fontsize=14)

    def copy(self):
        return copy.deepcopy(self)

    def minmaxvalue(self):
        minval = self.data[0][0]
        maxval = self.data[0][0]
        for i in range(self.size):
            #print i
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
        output = self.copy()
        for i in range(self.size):
            coord = output.getCoord(i)
            if output.getPixel(coord) > threshold:
                output.putPixel(coord, max)
            else:
                output.putPixel(coord, min)
        return output

    def invert(self):
        output = self.copy()
        for i in range(self.size):
            coord = output.getCoord(i)
            value = output.getPixel(coord)
            output.putPixel(coord, output.maxval - value)
        return output

    def normalize(self, inf, sup):
        output = self.copy()
        minval, maxval = output.minmaxvalue()
        output.maxval = sup
        a_value = (sup-inf) / ((maxval - minval) * 1.0)
        if minval < maxval:
            for p in range(output.size):
                coord = output.getCoord(p)
                result = ((a_value * output.getPixel(coord)) - (a_value * minval)) + inf
                output.data[coord[Y], coord[X]] = result
        else:
            self.error("Empty image", "normalize")
        output.data = output.data.astype("uint8")
        return output

    def logTransform(self):
        #To speed the process and avoid normalization, the c_value keeps
        #the result inside the [0, 255] range.
        output = self.copy()
        print output.minmaxvalue()
        c_value = 255/numpy.log10(255+1)
        output.data = numpy.log10(output.data + 2) * c_value
        print output.minmaxvalue()
        return output

    def expTransform(self):
        output = self.copy()
        c_value = 255 / math.e
        output.data = numpy.exp(output.data / 255.) * c_value
        print output.minmaxvalue()
        return output.normalize(0, 255)

    def squareTransform(self):
        output = self.copy()
        c_value = 255/(255*255)
        output.data = (output.data * output.data) * c_value
        return output

    def sqrtTransform(self):
        output = self.copy()
        c_value = 255/math.sqrt(255)
        output.data = numpy.sqrt(output.data) * c_value
        return output

    def constratTransform(self, alpha, beta, gamma):
        pass

'''
25 def _contrast_stretching(e, a, b, alpha, beta, gamma):
26     if 0 <= e <= a:
27         return alpha*e
28     if a < e <= b:
29         return beta*(e - a) + alpha*a
30     else:
31         return gamma*(e - b) + beta*(b - a) + alpha*a
32 
33 def contrast_stretching(img, params):
34     a = params['a']
35     b = params['b']
36     alpha = params['alpha']
37     beta = params['beta']
38     gamma = params['gamma']
39 
40     cs = numpy.vectorize(_contrast_stretching, excluded = ['a','b','alpha','beta','gamma'])
41     return cs(img, a, b, alpha, beta, gamma)'''
