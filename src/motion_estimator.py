import cv2
import math
import random
import numpy as np
from transformations import transform

__author__ = 'Marek'


class RansacMotionEstimator:
    def __init__(self, max_iterations, max_distance, hypothesis_set_length=10, min_inliers=None,
                 remember_inlier_indices=True):
        self.hypothesis_set_length = hypothesis_set_length
        self.max_iterations = max_iterations
        self.max_distance = max_distance
        self.min_inliers = min_inliers
        self.remember_inlier_indices = remember_inlier_indices

    def estimate(self, first_set, second_set):
        set_length = len(first_set)
        best_model = None
        best_model_inliers = 0
        inliers_indices = []
        range_set_length = range(set_length)

        for _ in xrange(self.max_iterations):
            if set_length < 3:
                continue
            random_indices = None
            if set_length > self.hypothesis_set_length:
                # This will return a list of hypothesis_set_length numbers selected from the range 0 to set_length, without duplicates.
                random_indices = random.sample(range_set_length, self.hypothesis_set_length)
            else:
                random_indices = range_set_length

            # create random subset
            first_subset = np.array([first_set[i] for i in random_indices], dtype=np.float32)
            second_subset = np.array([second_set[i] for i in random_indices], dtype=np.float32)

            # estimate model on random subset
            current_model = cv2.estimateRigidTransform(first_subset, second_subset, fullAffine=False)

            if current_model is None:
                continue

            current_model_inliers = 0
            current_inliers_indices = []

            # count inliers
            for index in xrange(set_length):
                transformed_point = transform(current_model, first_set[index][0])
                error = math.sqrt(
                    math.pow(transformed_point[0] - second_set[index][0][0], 2)
                    + math.pow(transformed_point[1] - second_set[index][0][1], 2)
                )

                if error < self.max_distance:
                    current_model_inliers += 1
                    if self.remember_inlier_indices:
                        current_inliers_indices.append(index)

            if current_model_inliers > best_model_inliers:
                best_model = current_model
                best_model_inliers = current_model_inliers
                inliers_indices = current_inliers_indices

        if best_model is None or (self.min_inliers is not None and best_model_inliers < self.min_inliers):
            best_model = cv2.estimateRigidTransform(first_set, second_set, fullAffine=False)
            if self.remember_inlier_indices:
                inliers_indices = [i for i in xrange(set_length)]

        return best_model, inliers_indices
