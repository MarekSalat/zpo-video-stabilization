__author__ = 'Marek'


class Filter:
    def __init__(self):
        pass

    def put(self, value):
        raise NotImplemented("Not implemented")

    def get(self):
        raise NotImplemented("Not implemented")


class KalmanFilter(Filter):
    def __init__(self, Q, R, one):
        self.estimation = None  # posteriori state estimate
        self.P = one  # posteriori estimate error covariance
        self.R = R  # measurement noise covariance
        self.Q = Q  # process noise covariance
        self.one = one

    def put(self, measured_value):
        if self.estimation is None:
            self.estimation = measured_value
            return self.estimation

        # Time Update (prediction)
        current_estimation = self.estimation
        _P = self.P + self.Q

        # Measurement Update (correction)
        kalman_gain = _P / (_P + self.R)
        new_estimation = current_estimation + kalman_gain * (measured_value - current_estimation)
        self.P = (self.one - kalman_gain) * _P

        # Update
        self.estimation = new_estimation
        return self.estimation

    def get(self):
        return self.estimation


