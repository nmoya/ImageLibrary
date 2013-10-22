#!/usr/bin/python
import libimg
import sys
import argparse

#http://wiki.scipy.org/Tentative_NumPy_Tutorial#\
#head-6a1bc005bd80e1b19f812e1e64e0d25d50f99fe2


def main():
    parser = argparse.ArgumentParser(description='Assignment #2')
    parser.add_argument('-t', '--task', help='Task [1]: Segment image',
                        required=False)
    parser.add_argument('-i', '--input', help='Input image', required=True)
    parser.add_argument('-o', '--output', help='Output image. result.pbm if \
                        not specified', required=False)

    args = vars(parser.parse_args())
    output_name = "result.pbm"
    if args["output"]:
        output_name = args["output"]

    print "Processing..."

    image = libimg.NMImage(filename=args["input"])
    #task = {"1": image.halfToningOrdered,
            #"2": image.halfToningDifuse()}

    output = image.morphologySegmentation()
    #image.show("Original")
    if output is not None:
        #output.show("Resultado")
        output.save(output_name)

    print "Result saved:", output_name
    image.showAll()

main()
