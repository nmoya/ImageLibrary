#!/usr/bin/python
import libimg
import sys
import argparse

#http://wiki.scipy.org/Tentative_NumPy_Tutorial#\
#head-6a1bc005bd80e1b19f812e1e64e0d25d50f99fe2


def main():
    parser = argparse.ArgumentParser(description='Assignment #1')
    parser.add_argument('-t', '--task', help='Task: [1,2,3,4,5]',
                        required=True)
    parser.add_argument('-i', '--input', help='Input image', required=True)
    parser.add_argument('-o', '--output', help='Output image. result.pgm if \
                        not specified', required=False)
    args = vars(parser.parse_args())
    output_name = "result.pgm"
    if args["output"]:
        output_name = args["output"]

    print "Processing..."

    image = libimg.NMImage(args["input"])
    task = {"1": image.logTransform,
            "2": image.expTransform,
            "3": image.squareTransform,
            "4": image.sqrtTransform,
            "5": image.constratTransform,
            "6": image.histogram}

    if args["task"] == "5":
        alpha = float(raw_input("Alpha:"))
        beta = float(raw_input("Beta:"))
        gamma = float(raw_input("Gamma:"))
        output = task[args["task"]](alpha, beta, gamma)
    else:
        output = task[args["task"]]()

    image.histogram(title="Input Histogram")
    output.histogram(title="Output histogram")
    image.show("Original")
    output.show("Transformada")
    output.save(output_name)

    print "Result saved:", output_name
    output.showAll()

main()
