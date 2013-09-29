#!/usr/bin/python
import libimg
import sys
import argparse

#http://wiki.scipy.org/Tentative_NumPy_Tutorial#\
#head-6a1bc005bd80e1b19f812e1e64e0d25d50f99fe2


def main():
    parser = argparse.ArgumentParser(description='Assignment #2')
    parser.add_argument('-t', '--task', help='Task: [1,2]: 1-Ordered Half \
                        Toning; 2-Difused Half Toning', required=True)
    parser.add_argument('-i', '--input', help='Input image', required=True)
    parser.add_argument('-o', '--output', help='Output image. result.pgm if \
                        not specified', required=False)
    parser.add_argument('-s', '--scan', help='1: Regular, 2: ZigZag',
                        required=True)

    args = vars(parser.parse_args())
    output_name = "result.pgm"
    if args["output"]:
        output_name = args["output"]

    print "Processing..."

    image = libimg.NMImage(filename=args["input"])
    #task = {"1": image.halfToningOrdered,
            #"2": image.halfToningDifuse()}

    if args["task"] == "1":
        output = image.halfToningOrdered()
    else:
        output = image.halfToningDifuse(args["scan"])

    image.show("Original")
    output.show("Resultado")
    output.save(output_name)

    print "Result saved:", output_name
    image.showAll()

main()
