import numpy as np


def load_data():

    def format_ret(line):
        """

        :param line:
        :type line: str
        :return:
        :rtype: []
        """
        return [float(value) for value in line.replace("\n", "").split(",")]

    ret = []
    with open("./data.txt", mode="r") as file:
        ret = [format_ret(line) for line in file.readlines()]
    return ret

