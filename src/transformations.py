import math
import numpy as np

__author__ = 'Marek'


def get_x(mat):
    return mat[0, 2]


def get_y(mat):
    return mat[1, 2]


def get_rad_angle(mat):
    return math.atan2(mat[1, 0], mat[0, 0])


def fill_mat(mat, dx, dy, angle):
    mat[0, 0] = math.cos(angle)
    mat[0, 1] = -math.sin(angle)
    mat[1, 0] = math.sin(angle)
    mat[1, 1] = math.cos(angle)

    mat[0, 2] = dx
    mat[1, 2] = dy


def transform(mat, point):
    return (
        mat[0, 0] * point[0] + mat[0, 1] * point[1] + mat[0, 2],
        mat[1, 0] * point[0] + mat[1, 1] * point[1] + mat[1, 2]
    )


