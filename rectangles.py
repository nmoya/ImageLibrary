class Rectangle():
    """docstring for Rectangle"""
    def __init__(self, pos_x, pos_y, width, height, pixels, black,
                 vertical_trans, horizontal_trans, area, label):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width
        self.height = height
        self.pixels = pixels
        self.black_trans = black
        self.vertical_trans = vertical_trans
        self.horizontal_ratio = horizontal_trans
        self.area = area
        self.label = label

    def __repr__(self):
        return "Label: ", self.label, "Black ratio: ", self.black_ratio, \
            "Vertical Ratio: ", self.vertical_ratio
