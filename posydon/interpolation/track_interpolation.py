""" Module for performing track interpolation """

__authors__ = [
    "Philipp Moura Srivastava <philipp.msrivastava@northwestern.edu>"
]

from posydon.interpolation.IF_interpolation import IFInterpolator
from posydon.utils.common_functions import PATH_TO_POSYDON_DATA
from posydon.grids.downsampling import TrackDownsampler
from posydon.interpolation.utils import set_valid
from posydon.grids.psygrid import PSyGrid
from scipy.interpolate import interp1d
import numpy as np
import os


class TrackInterpolator:

    def __init__(self, in_keys, out_keys, method, phase, n_points = 9):

        self.in_keys = in_keys
        self.out_keys = out_keys # should always include age to get evaluation to work

        methods = {
            "1NN": self.__test_1nn,
            "DSCP": self.__test_changepoints_knn,
            "kNN": self.__test_average_knn
        }

        self.method = methods[method]
        self.phase = phase
        self.n_points = n_points


    def load(self):
        pass

    def train(self, grid):

        self.tracks = []

        self.iv = np.array(grid.initial_values[self.in_keys].tolist())
        self.classes = grid.final_values["interpolation_class"]
        unique_classes = np.unique(self.classes)

        valid_inds = set_valid(self.classes, self.iv, unique_classes,
                    self.phase == "HMS-HMS")

        self.iv = self.iv[valid_inds >= 0] # initial values

        self.classes = self.classes[valid_inds >= 0]

        for ind, v in enumerate(valid_inds):
            if v >= 0:
                bin_hist = np.array(grid[ind].binary_history[self.out_keys].tolist())
                self.tracks.append(bin_hist)

        self.tracks = np.array(self.tracks)

        IFInterp = IFInterpolator()

        IFInterp.load(os.path.join(PATH_TO_POSYDON_DATA, "POSYDON_data", self.phase,
                                    "interpolators", "linear3c_kNN", "grid_0.0142.pkl"))

        self.classifier = IFInterp.interpolators[0]

    def test_interpolator(self, initial_values):

        new_values = np.array(initial_values, copy = True)

        if self.phase == "HMS-HMS":
            new_values.T[1] = new_values.T[1] / new_values.T[0]

        classes = self.classifier.test_classifier("interpolation_class", new_values)

        approx = []

        for _class, iv in zip(classes, initial_values):
            approx.extend(self.method(iv, _class))

        return np.array(approx)

    def evaluate(self, binary):
        pass

    def save(self):
        pass

    def __test_1nn(self, initial_value, class_label):
        return self.__k_nearest(initial_value, 1, class_label)

    def __test_changepoints_knn(self, initial_value, class_label):
        
        k_nearest = self.__k_nearest(initial_value, 4, class_label)

        cps = []

        for nearest in k_nearest:
            cps.append(self.__find_changepoints(nearest))

        return [np.array(cps).mean(axis = 0)]

    def __test_average_knn(self, initial_value, class_label):

        k_nearest = self.__k_nearest(initial_value, 4, class_label)

        avg = []

        for nearest in k_nearest:

            if nearest.shape[0] < 2:
                continue
            
            norm_age = (nearest.T[0] - nearest.T[0].min()) / (nearest.T[0].max() - nearest.T[0].min() + 1.0e-8)

            interp = interp1d(norm_age, nearest.T[1:])

            dt = 1 / 1000

            age = 0

            temp = []

            while age < 1.0:

                temp.append([age, interp(age)])

                age += dt

            avg.append(temp)

        avg = np.array(avg).mean(axis = 0)

        return [avg]

    def __find_changepoints(self, track):

        subset = [track.T[1:].T[0], track.T[1:].T[-1]]
        age = [track.T[0][0], track.T[0][-1]]

        for i in range(self.n_points):
            order = np.array(age).argsort()
            interp = interp1d(np.array(age)[order], np.array(subset)[order].T)

            max_err = -np.inf
            age_to_add = age[0]
            sub_to_add = subset[0]

            for tb in track:
                err = np.abs(tb[1:] - interp(tb[0])).sum()
                if err > max_err:
                    max_err = err
                    age_to_add =  tb[0]
                    sub_to_add = tb[1:]

            age.append(age_to_add)
            subset.append(sub_to_add)

        ret = np.vstack([age, np.array(subset).T])

        return ret.T[ret[0].argsort()]


    def __k_nearest(self, initial_value, k, class_label):

        class_inds = np.where(self.classes == class_label)[0]

        return self.tracks[class_inds[np.sqrt(np.sum(np.square(self.iv[class_inds] - initial_value), axis = 1)).argsort()[:k]]]
    