__author__ = 'Marek'


class Trajectory:
    def __init__(self, x=0., y=0., angle=0.):
        self.x = x
        self.y = y
        self.angle = angle

    def __add__(self, other):
        return Trajectory(self.x + other.x, self.y + other.y, self.angle + other.angle)

    def __sub__(self, other):
        return Trajectory(self.x - other.x, self.y - other.y, self.angle - other.angle)

    def __mul__(self, other):
        return Trajectory(self.x * other.x, self.y * other.y, self.angle * other.angle)

    def __div__(self, other):
        return Trajectory(self.x / other.x, self.y / other.y, self.angle / other.angle)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.angle == other.angle

    def __repr__(self):
        return "<class '%s' [x: %f, y:%f, angle:%f]>" % (self.__class__.__name__, self.x, self.y, self.angle)