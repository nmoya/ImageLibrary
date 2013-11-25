#!/usr/bin/python
import libimg
import sys
import argparse

#http://wiki.scipy.org/Tentative_NumPy_Tutorial#\
#head-6a1bc005bd80e1b19f812e1e64e0d25d50f99fe2


def main():
    parser = argparse.ArgumentParser(description='Assignment #4')
    parser.add_argument('-a', '--angle', help='Angle (ccw)', required=False)
    parser.add_argument('-s', '--scale', help='Scale Factor', required=False)
    parser.add_argument('-d', '--dimension', help='WxH', required=False)
    parser.add_argument('-m', '--method', help='Scale Method', required=True)
    parser.add_argument('-i', '--input', help='Input image', required=True)
    parser.add_argument('-o', '--output', help='Output image. result.ppm if \
                        not specified', required=False)

    args = vars(parser.parse_args())
    output_name = "result.ppm"
    if args["output"] is not None:
        output_name = args["output"]

    print "Processing..."

    image = libimg.NMImage(filename=args["input"])
    print "Original: ", image.sizex, image.sizey, image.size
    #image.show("Original")
    #output = image.rgb_to_hsv()
    #output.show("Converted")

    scaled_image = None
    rotated_image = None

    if args["scale"] is not None:
        scaled_image = image.scale(args["scale"], args["method"])
        print "Scaled: ", scaled_image.sizex, scaled_image.sizey, scaled_image.size
    elif args["dimension"] is not None:
        scaled_image = image.scale(args["dimension"].split("x"),
                                   args["method"])

    if args["angle"] is not None:
        rotated_image = image.rotate(args["angle"])

    if scaled_image is not None:
        scaled_image.save(output_name)
    if rotated_image is not None:
        rotated_image.save(output_name)

    print "Result saved:", output_name
    image.showAll()

main()
