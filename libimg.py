import skimage.io as io
import numpy
import re
import math
import copy

X = 0
Y = 1


class NMImage ():
    def __init__(self, **kwargs):
        self.filename = kwargs.get("filename", '')
        self.sizex = kwargs.get("sizex", 0)
        self.sizey = kwargs.get("sizey", 0)
        self.maxval = 255
        if self.filename == '':
            self.data = numpy.zeros(self.sizex * self.sizey)
            self.data = self.data.astype("float64")
            self.data = self.data.reshape(self.sizex, self.sizey)
        else:
            self.data = self.open()
        self.size = (self.sizey * self.sizex)

    def __repr__(self):
        name = self.filename.split("/")[-1]
        return name + " (" + str(self.sizex) + ", " + str(self.sizey) + ")"

    def open(self):
        #http://flockhart.virtualave.net/RBIF0100/regexp.html
        io.use_plugin("freeimage")
        self.data = io.imread(self.filename)
        self.data = self.data.astype("float64")
        self.sizex = self.data.shape[0]
        self.sizey = self.data.shape[1]

        return self.data

    def save(self, name):
        if self.data.dtype != "uint8":
            self.data = self.data.astype("uint8")
        io.use_plugin("freeimage")
        io.imsave(name, self.data)

    def getPixel(self, coord):
        return self.data[coord[X], coord[Y]]

    def putPixel(self, coord, value):
        self.data[coord[X], coord[Y]] = value

    def getCoord(self, position):
        #http://cl.ly/image/380z3r3y1H19
        x = (position % (self.size)) % self.sizex
        y = (position % (self.size)) / self.sizex
        return (x, y)

    def show(self, title=''):
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        plt.figure()
        if self.filename.endswith("pgm") or len(self.filename) == 0:
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
        plt.hist(self.data.reshape(-1), bins=256)
        plt.xlim([0, 255])
        if len(title) == 0:
            title = self.filename
        plt.title(title, fontsize=14)

    def copy(self):
        return copy.deepcopy(self)

    def validCoord(self, coord):
        x = coord[X]
        y = coord[Y]
        if (x >= 0) and (x < self.sizex) and \
           (y >= 0) and (y < self.sizey):
            return True
        else:
            return False

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
        a_value = (sup-inf) / ((maxval * 1.0 - minval * 1.0) * 1.0)
        if minval < maxval:
            for p in range(output.size):
                coord = output.getCoord(p)
                result = ((a_value * output.getPixel(coord)) -
                         (a_value * minval)) + inf
                output.putPixel(coord, result)
        else:
            self.error("Empty image", "normalize")
        return output

    def halfToningOrdered(self):
        normalized_img = self.normalize(0, 9)
        output = NMImage(sizex=self.sizex*3, sizey=self.sizey*3)

        threshold_matrix = [[0, 0, 0], [-1, 0, 1], [0, 1, 2],
                            [1, 0, 3], [1, -1, 4], [-1, 1, 5],
                            [-1, -1, 6], [1, 1, 7], [0, -1, 8]]

        for i in range(self.size):
            coord = normalized_img.getCoord(i)
            value = normalized_img.getPixel(coord)

            for neighbour in threshold_matrix:
                new_coord = (coord[X] + neighbour[X], coord[Y] + neighbour[Y])
                new_coord = (new_coord[0]*3, new_coord[1]*3)
                if output.validCoord(new_coord):
                    if value <= neighbour[2]:
                        output.putPixel(new_coord, 0)
                    else:
                        output.putPixel(new_coord, 255)
        return output

    def halfToningDifuse(self, zigzag):
        output = NMImage(sizex=self.sizex, sizey=self.sizey)
        copy = self.copy()
        neighbourhood = [[1, 0, 0.435], [-1, -1, 0.1875],
                         [0, 1, 0.3125], [1, 1, 0.0625]]

        for y in range(output.sizey):
            x_interval = range(0, self.sizex)
            if y % 2 == 1 and zigzag == 2:
                x_interval = reversed(x_interval)

            for x in x_interval:
                original_value = copy.getPixel((x, y))
                error = 0
                if original_value > 127:
                    output.putPixel((x, y), 255.)
                    error = original_value - 255.
                else:
                    output.putPixel((x, y), 0.)
                    error = original_value

                for neighbour in neighbourhood:
                    new_coord = (x + neighbour[X], y + neighbour[Y])
                    if output.validCoord(new_coord):
                        coef = neighbour[2]
                        previous_value = copy.getPixel(new_coord)
                        copy.putPixel(new_coord,
                                      (previous_value + (coef * error)))

        return output

    def logTransform(self):
        #To speed the process and avoid normalization, the c_value keeps
        #the result inside the [0, 255] range.
        output = self.copy()
        c_value = 255/numpy.log10(255.+1)
        output.data = output.data + 1.
        output.data = numpy.log10(output.data) * c_value
        return output

    def expTransform(self):
        output = self.copy()
        c_value = 255 / math.e
        output.data = numpy.exp(output.data / 255.) * c_value
        return output.normalize(0, 255)

    def squareTransform(self):
        output = self.copy()
        c_value = 255./(255. * 255.)
        output.data = numpy.square(output.data) * c_value
        return output.normalize(0, 255)

    def sqrtTransform(self):
        output = self.copy()
        c_value = 255/math.sqrt(255.)
        output.data = numpy.sqrt(output.data) * c_value
        return output

    def constratTransform(self, a, b, alpha, beta, gamma):
        output = self.copy()
        minval, maxval = self.minmaxvalue()
        for p in range(output.size):
            coord = output.getCoord(p)
            value = output.getPixel(coord)
            if value >= 0 and value <= a:
                value = alpha * value
            elif a < value and value <= b:
                value = beta * (value - a) + (alpha * a)
            elif b < value and value <= maxval:
                value = gamma * (value - b) + beta * (b - a) + (alpha * a)
            output.putPixel(coord, value)
        return output.normalize(0, 255)
