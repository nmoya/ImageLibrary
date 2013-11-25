from __future__ import division
from skimage.measure import regionprops
from skimage.color import label2rgb, colorconv
from matplotlib import pyplot as plt
from rectangles import *
from scipy import ndimage
import pylab
import matplotlib.patches as mpatches
import matplotlib.cm as cm
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
        self.color = kwargs.get("color", False)
        self.maxval = 255
        if self.filename == '':
            if self.color:
                self.data = numpy.zeros(self.sizex * self.sizey * 3)
                self.data = self.data.reshape(self.sizex, self.sizey, 3)
            else:
                self.data = numpy.zeros(self.sizex * self.sizey)
                self.data = self.data.reshape(self.sizex, self.sizey)
        else:
            self.data = self.open()

        self.size = self.sizex * self.sizey
        self.data = self.data.astype("float64")

    def __repr__(self):
        name = self.filename.split("/")[-1]
        return name + " (" + str(self.sizex) + ", " + str(self.sizey) + ")"

    def open(self):
        #http://flockhart.virtualave.net/RBIF0100/regexp.html
        if not self.filename.endswith(".pbm"):
            io.use_plugin("freeimage")
        self.data = io.imread(self.filename)
        #self.data = self.data.astype("float64")
        self.sizex = self.data.shape[0]
        self.sizey = self.data.shape[1]
        if self.filename.endswith("ppm"):
            self.color = True

        return self.data

    def save(self, name):
        if self.data.dtype != "uint8":
            self.data = self.data.astype("uint8")
        io.use_plugin("freeimage")
        print "Saving: ", self.sizex, self.sizey
        io.imsave(name, self.data)

    def getPixel(self, coord):
        return self.data[coord[X], coord[Y]]

    def putPixel(self, coord, value):
        self.data[coord[X], coord[Y]] = value

    def getCoord(self, position):
        #http://cl.ly/image/380z3r3y1H19
        x = int((position % (self.size)) % self.sizex)
        y = int((position % (self.size)) / self.sizex)
        return (x, y)

    def show(self, title=''):
        plt.figure()
        if self.filename.endswith("pgm") or self.filename.endswith("pbm") \
           or len(self.filename) == 0:
            plt.imshow(self.data, cmap=cm.gray)
        elif self.filename.endswith("ppm"):
            plt.imshow(self.data, cmap=cm.hsv)
        else:
            plt.imshow(self.data)
        plt.axis('off')
        if len(title) == 0:
            title = self.filename
        plt.title(title, fontsize=14)

    def showBBoxes(self, rectangleList):
        COLORS = ["yellow", "red"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.data, cmap=cm.gray)
        counter = 0
        for r in rectangleList:
            if r.label == 1:
                counter += 1
                rect = mpatches.Rectangle((r.pos_x, r.pos_y), r.width,
                                          r.height, fill=False,
                                          edgecolor=COLORS[r.label],
                                          linewidth=2)
            ax.add_patch(rect)
        fig.savefig("result"+str(counter)+".png")

    def showLabelImage(self, labelImage):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(label2rgb(labelImage.data, image=self.data))

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
        if self.color:
            print "Not impelented minmax for color images"
            return 0, 0
        else:
            return numpy.amin(self.data), numpy.amax(self.data)

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

   def scaleMatrix(self, scale_x, scale_y):
        matrix = numpy.zeros(9)
        matrix = matrix.reshape((3, 3))

        matrix[0, 0] = scale_x
        matrix[1, 1] = scale_y
        matrix[2, 2] = 1  # W

        return matrix

    def rotationMatrix(self, theta):
        matrix = numpy.zeros(9)
        matrix = matrix.reshape((3, 3))

        matrix[0, 0] = math.cos(theta)
        matrix[1, 1] = math.cos(theta)

        matrix[0, 1] = - math.sin(theta)
        matrix[1, 0] = math.sin(theta)

        matrix[2, 2] = 1  # W

        return matrix

    def computeOffset(self, direct_matrix):
        edge = numpy.array([[0, 0, 1],
                            [self.sizex-1, 0, 1],
                            [self.sizex-1, self.sizey-1, 1],
                            [0, self.sizey-1, 1]])
        edge = direct_matrix.dot(edge.T)
        return edge[0].min(), edge[1].min()

    def interpolate_nneighbour(self, output_image, inverted_matrix,
                               direct_matrix=None):
        xmin, ymin = 0, 0
        if direct_matrix is not None:
            xmin, ymin = self.computeOffset(direct_matrix)

        for i in range(output_image.size):
            current_coord = output_image.getCoord(i)
            transformed_coord = inverted_matrix.dot([current_coord[X]+xmin,
                                                    current_coord[Y]+ymin, 1])

            transformed_coord = (int(transformed_coord[X]),
                                 int(transformed_coord[Y]))

            if self.validCoord(transformed_coord):  # Valid coord?
                output_image.putPixel(current_coord,
                                      self.getPixel(transformed_coord))

        return output_image

    def interpolate_bilinear(self, output_image, inverted_matrix,
                             direct_matrix=None):
        extended_image = numpy.pad(self.data, ((0, 1), (0, 1), ((0, 0))),
                                   "edge")

        xmin, ymin = 0, 0
        if direct_matrix is not None:
            xmin, ymin = self.computeOffset(direct_matrix)

        for i in range(output_image.size):
            current_coord = output_image.getCoord(i)
            transformed_coord = inverted_matrix.dot([current_coord[X]+xmin,
                                                    current_coord[Y]+ymin, 1])

            rounded_coord = (int(transformed_coord[X]),
                             int(transformed_coord[Y]))

            dx = transformed_coord[X] - rounded_coord[X]
            dy = transformed_coord[Y] - rounded_coord[Y]

            value = 0
            if self.validCoord(rounded_coord):
                value = (1-dx)*(1-dy)*extended_image[rounded_coord[X], rounded_coord[Y]] + \
                    dx*(1-dy)*extended_image[rounded_coord[X]+1, rounded_coord[Y]] +      \
                    (1-dx)*dy*extended_image[rounded_coord[X], rounded_coord[Y]+1] +      \
                    dx*dy*extended_image[rounded_coord[X]+1, rounded_coord[Y]+1]
                output_image.putPixel(current_coord, value)

        return output_image

    def R(self, s):
        value = self.P(s+2)**3 - 4*self.P(s+1)**3 + \
            6*self.P(s)**3 - 4*self.P(s-1)**3
        value = value/6.
        return value

    def P(self, t):
        if t > 0:
            return t
        else:
            return 0

    def interpolate_bicubic(self, output_image, inverted_matrix,
                            direct_matrix=None):
        xmin, ymin = 0, 0
        if direct_matrix is not None:
            xmin, ymin = self.computeOffset(direct_matrix)
        extended_image = numpy.pad(self.data, ((1, 2), (1, 2), ((0, 0))),
                                   "edge")

        neighbours = []
        for i in range(-1, 3):
            for j in range(-1, 3):
                neighbours.append((i, j))

        for i in range(output_image.size):
            current_coord = output_image.getCoord(i)
            transformed_coord = inverted_matrix.dot([current_coord[X]+xmin,
                                                    current_coord[Y]+ymin, 1])

            rounded_coord = (int(transformed_coord[X]),
                             int(transformed_coord[Y]))

            dx = transformed_coord[X] - rounded_coord[X]
            dy = transformed_coord[Y] - rounded_coord[Y]

            value = 0
            for neigh in neighbours:
                n_coord = (transformed_coord[X] + neigh[X],
                           transformed_coord[Y] + neigh[Y])
                if self.validCoord(n_coord):  # Valid coord?
                    value += self.getPixel(n_coord) * \
                        self.R(neigh[X]-dx) * self.R(dy-neigh[Y])
            output_image.putPixel(current_coord, value)

        return output_image

    def scale(self, scale_factor, method):
        if isinstance(scale_factor, list):
            scale_factor = map(int, scale_factor)
            scaled_image = NMImage(sizex=scale_factor[0],
                                   sizey=scale_factor[1],
                                   color=True)
            scale_x = float(scale_factor[0]/self.sizex)
            scale_y = float(scale_factor[1]/self.sizey)
        else:
            scale_x = float(scale_factor)
            scale_y = float(scale_factor)
            scaled_image = NMImage(sizex=int(self.sizex*scale_x),
                                   sizey=int(self.sizey*scale_y),
                                   color=True)

        inverted_matrix = self.scaleMatrix(1./scale_x, 1./scale_y)

        if method == 'nneighbour':
            scaled_image = self.interpolate_nneighbour(scaled_image,
                                                       inverted_matrix)
        elif method == 'bilinear':
            scaled_image = self.interpolate_bilinear(scaled_image,
                                                     inverted_matrix)
        elif method == 'bicubic':
            scaled_image = self.interpolate_bicubic(scaled_image,
                                                    inverted_matrix)
        return scaled_image

    def rotate(self, theta, method):
        theta = math.radians(theta)
        new_x = math.fabs(self.sizex*math.cos(theta))+math.fabs(self.sizey*math.sin(theta))
        new_y = math.fabs(self.sizex*math.sin(theta))+math.fabs(self.sizey*math.cos(theta))
        rotated_image = NMImage(sizex=int(new_x), sizey=int(new_y), color=True)
        #rotated_image = NMImage(sizex=int(self.sizex), sizey=int(self.sizey), color=True)
        
        direct_matrix = self.rotationMatrix(theta)
        inverted_matrix = self.rotationMatrix(-theta)

        if method == 'nneighbour':
            rotated_image = self.interpolate_nneighbour(rotated_image,
                                                        inverted_matrix,
                                                        direct_matrix)

        elif method == 'bilinear':
            rotated_image = self.interpolate_bilinear(rotated_image,
                                                      inverted_matrix,
                                                      direct_matrix)
        elif method == 'bicubic':
            rotated_image = self.interpolate_bicubic(rotated_image,
                                                     inverted_matrix,
                                                     direct_matrix)
        return rotated_image

    def erode(self, kernel):
        kernel_sum = kernel.sum()
        output = NMImage(sizex=self.sizex, sizy=self.sizey)
        convolved = numpy.empty_like(self.data, dtype=numpy.uint)

        convolved = ndimage.convolve(self.data, kernel, mode='constant',
                                     cval=1)
        output.data = numpy.equal(convolved, kernel_sum)
        return output

    def dilation(self, kernel):
        output = NMImage(sizex=self.sizex, sizy=self.sizey)
        convolved = numpy.empty_like(self.data, dtype=numpy.uint)
        convolved = ndimage.convolve(self.data, kernel, mode='constant',
                                     cval=0)
        output.data = numpy.not_equal(convolved, 0)
        return output

    def closing(self, kernel):
        dilatated = self.dilation(kernel)
        return dilatated.erode(kernel)

    def opening(self, kernel):
        eroded = self.erode(kernel)
        return eroded.dilation(kernel)

    def morphElement(self, width, height):
        return morphology.rectangle(width, height)

    def morphologySegmentation(self):
        morphological_elements = [[(100, 1), (1, 200), (1, 30), "Lines"],
                                  [(30, 1), (1, 20), (1, 6), "Words"],
                                  [(30, 1), (1, 20), (1, 1), "Letters"]]
        for m_element in morphological_elements:
            rectangleList = []
            step12 = self.binarize()
            element12 = self.morphElement(m_element[0][0], m_element[0][1])
            step12 = step12.closing(element12)
            #step12.show("Step 1-2")

            step34 = self.binarize()
            element34 = self.morphElement(m_element[1][0], m_element[1][1])
            step34 = step34.closing(element34)
            #step34.show("Step 3-4")

            step5 = self.copy()
            step5.data = numpy.logical_and(step12.data, step34.data)
            #step5.show("Step 5")

            step6 = step5.copy()
            step6.show("Letras")
            element6 = self.morphElement(m_element[2][0], m_element[2][1])
            step6 = step6.closing(element6)
            step6.show("Step 6")

            labels = step6.copy()
            #labels.data = labels.data.astype("int")
            labels.data = morphology.label(labels.data, 8, 0)
            #labels.show("Labels")
            #print "Number of Labels: ", labels.data.max()

            #Retrieve the retangles from the label image
            binary_input = self.binarize()
            for region in regionprops(labels.data, ['BoundingBox', "label"]):

                rect_y, rect_x, rect_y_2, rect_x_2 = region['BoundingBox']
                width = rect_x_2 - rect_x
                height = rect_y_2 - rect_y
                patch = binary_input.data[rect_y:rect_y_2+1, rect_x:rect_x_2+1]

                horizontal, vertical, previous = 0, 0, 0
                area, black = patch.size, patch.sum()

                for lin in range(len(patch)):
                    for col in range(len(patch[0])):
                        if patch[lin, col] != previous:
                            horizontal += 1
                        previous = patch[lin, col]
                previous = 0
                for col in range(len(patch[0])):
                    for lin in range(len(patch)):
                        if patch[lin, col] != previous:
                            vertical += 1
                        previous = patch[lin, col]

                black_ratio = black / area
                horizontal_ratio = horizontal / area
                vertical_ratio = vertical / area

                if (black_ratio > 0.18 and black_ratio < 0.75 and area > 100) \
                    or math.fabs(horizontal_ratio-vertical_ratio) > 0.005 \
                    and horizontal_ratio > vertical_ratio \
                   and area > 100:
                    rectangleList.append(Rectangle(rect_x, rect_y, width,
                                         height, patch, black, vertical,
                                         horizontal, area, 1))
                else:
                    rectangleList.append(Rectangle(rect_x, rect_y, width,
                                         height, patch, black, vertical,
                                         horizontal, area, 0))
                    #print black / area, horizontal / area, vertical/area, area
            print m_element[3], len([r for r in rectangleList if r.label == 1])
            self.showBBoxes(rectangleList)

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
        output.maxval = self.data.max()
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
