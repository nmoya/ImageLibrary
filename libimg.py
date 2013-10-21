import skimage.io as io
import skimage.morphology as morphology
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
        if not self.filename.endswith(".pbm"):
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
#                                                     #
#                                                     #
#                   NOT IO                            #
#                                                     #
#                                                     #
#######################################################

    def halfToningOrdered(self):
        normalized_img = self.normalize(0, 9)
        output = NMImage(sizex=self.sizex*3, sizey=self.sizey*3)

        threshold_matrix = [[0, 0, 0], [-1, 0, 1], [0, 1, 2],
                            [1, 0, 3], [1, -1, 4], [-1, 1, 5],
                            [-1, -1, 6], [1, 1, 7], [0, -1, 8]]

        for y in range(self.sizey):
            for x in range(self.sizex):
                coord = (x, y)
                value = normalized_img.getPixel(coord)

                counter = 0
                for new_y in range(y*3, (y*3)+3):
                    for new_x in range(x*3, (x*3)+3):
                        neighbour = threshold_matrix[counter]
                        new_coord = (new_x + neighbour[X],
                                     new_y + neighbour[Y])
                        if output.validCoord(new_coord):
                            if value <= neighbour[2]:
                                output.putPixel(new_coord, 0.)
                            else:
                                output.putPixel(new_coord, 255.)
                        counter += 1
        return output

    def halfToningDifuse(self, zigzag):
        output = NMImage(sizex=self.sizex, sizey=self.sizey)
        copy = self.copy()
        neighbourhood = [[1, 0, 0.435], [-1, -1, 0.1875],
                         [0, 1, 0.3125], [1, 1, 0.0625]]

        for y in range(output.sizey):
            x_interval = range(0, self.sizex)
            if y % 2 == 1 and zigzag == "2":
                x_interval = reversed(x_interval)
                print x_interval

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

    def morphElement(self, size):
        sizex = size[0]
        sizey = size[1]
        return numpy.reshape(numpy.array([1 for i in range(sizex*sizey)]),
                             (sizex, sizey))

    def morphologySegmentation(self):
        step12 = self.binarize()
        step12 = step12.invert()
        element12 = self.morphElement((100, 1))
        step12.data = morphology.binary_opening(self.data, element12)
        step12.show("Step 1-2")

        step34 = self.binarize()
        element34 = self.morphElement((1, 200))
        step34.data = morphology.binary_opening(self.data, element34)
        step34.show("Step 3-4")

        step5 = self.copy()
        for i in range(step5.size):
            coord = step5.getCoord(i)
            value12 = step12.getPixel(coord)
            value34 = step34.
            getPixel(coord)
            step5.putPixel(coord, (value12 and value34))
        step5.show("Step 5")

        element6 = self.morphElement((30, 1))
        step6 = self.copy()
        step6.data = morphology.binary_closing(step5.data, element34)
        step6.show("Step 6")

        step6.data = step6.data.astype("int")
        labels = self.copy()
        labels.data = labels.data.astype("int")
        labels.data = morphology.label(step6.data, 4, 0)
        labels.normalize(0, 255).show("Normalized labels")

        #Retrieve the retangles from the label image

        return step5

#######################################################
#                                                     #
#                                                     #
#                   Transformations                   #
#                                                     #
#                                                     #
#######################################################

    def binarize(self):
        output = self.copy()
        output.data = (output.data == 0).astype("uint8")
        output.maxval = 1
        return output

    def threshold(self, threshold, min, max):
        output = self.copy()
        for i in range(self.size):
            coord = output.getCoord(i)
            if output.getPixel(coord) > threshold:
                output.putPixel(coord, max)
            else:
                output.putPixel(coord, min)
        output.maxval = max
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
