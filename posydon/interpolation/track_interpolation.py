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

    def __init__(self, in_keys, out_keys, method, phase, n_points = 9, max_interp = None):

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

        if max_interp is not None:
            self.max_interp = IFInterpolator()

            self.max_interp.load(max_interp)
        else:
            self.max_interp = None



    def load(self):
        pass

    def train(self, grid):

        self.tracks = []

        self.iv = np.array(grid.initial_values[self.in_keys].tolist())
        self.classes = grid.final_values["interpolation_class"]
        unique_classes = np.unique(self.classes)

        valid_inds = set_valid(self.classes, self.iv, unique_classes,
                    self.phase == "HMS-HMS")

        # taking out systems that start in RLOF
        valid_inds[np.where(grid.final_values["termination_flag_1"] == "forced_initial_RLO")[0]] = -1
        valid_bool = lambda track: track.history1 is None or track.history2 is None or track.binary_history is None if self.phase == "HMS-HMS" else track.history1 is None or track.binary_history is None
        valid_inds[[ind for ind, track in enumerate(grid) if valid_bool(track)]] = -1


        self.iv = self.iv[valid_inds >= 0] # initial values
        self.iv_max, self.iv_min = self.iv.max(axis = 0), self.iv.min(axis = 0)
        
        self.iv = (self.iv - self.iv_min) / (self.iv_max - self.iv_min)

        self.classes = self.classes[valid_inds >= 0]

        bhist_keys = [k for k in self.out_keys if k.split("_")[0] != "S1" and k.split("_")[0] != "S2"]
        s1_keys = [k[3:] for k in self.out_keys if k.split("_")[0] == "S1"]
        s2_keys = [k[3:] for k in self.out_keys if k.split("_")[0] == "S2"]

        if self.phase != "HMS-HMS" and len(s2_keys) > 0:
            # self.out_keys = [k for k in self.out_keys if k[3:] not in s2_keys] + s1_keys
            print("Star 2 is undefined for grids containing CO")

        for ind, v in enumerate(valid_inds):
            if v >= 0:
                
                if self.phase == "HMS-HMS":
                    bin_hist = np.array(grid[ind].binary_history[bhist_keys].tolist())
                    s1_hist = np.array(grid[ind].history1[s1_keys].tolist())
                    s2_hist = np.array(grid[ind].history2[s2_keys].tolist())

                    self.tracks.append(np.concatenate([bin_hist, s1_hist, s2_hist], axis = 1))
                else:
                    bin_hist = np.array(grid[ind].binary_history[bhist_keys].tolist())
                    s1_hist = np.array(grid[ind].history1[s1_keys].tolist())

                    self.tracks.append(np.concatenate([bin_hist, s1_hist], axis = 1))

        self.tracks = np.array(self.tracks)


        IFInterp = IFInterpolator()

        IFInterp.load(os.path.join(PATH_TO_POSYDON_DATA, "POSYDON_data", self.phase,
                                    "interpolators", "linear3c_kNN", "grid_0.0142.pkl"))

        self.classifier = IFInterp.interpolators[0]

    def test_interpolator(self, initial_values, meta = False):

        new_values = np.array(initial_values, copy = True)

        if self.phase == "HMS-HMS":
            new_values.T[1] = new_values.T[1] / new_values.T[0]

        classes = self.classifier.test_classifier("interpolation_class", new_values)

        approx = []
        meta_data = []

        for _class, iv in zip(classes, initial_values):
            if meta:
                a, m = self.method(iv, _class, meta)

                approx.extend(a)
                meta_data.append(m)

            else:
                approx.extend(self.method(iv, _class))

        return np.array(approx), np.array(meta_data) if meta else np.array(approx)

    def evaluate(self, binary):
        pass

    def save(self):
        pass

    def __test_1nn(self, initial_value, class_label, meta = False):
        nearest = self.__k_nearest(initial_value, 1, class_label)[0]

        if self.max_interp is not None:
            y_max = self.max_interp.test_interpolator(np.array([initial_value]))[0]
            # rescaling y using approx max
            nearest[0].T[1:] = ((nearest[0].T[1:].T / nearest[0].max(axis = 0)[1:]) * y_max).T

        if meta:
            nearest = nearest, [nearest] * 4

        return nearest

    def __test_changepoints_knn(self, initial_value, class_label, get_neighbors = False):
        
        k_nearest, distances = self.__k_nearest(initial_value, 4, class_label)

        weights = 1 / np.power(distances, 2)

        cps = []

        for nearest in k_nearest:
            cps.append(self.__find_changepoints(nearest)[0])

        app = np.average(np.array(cps), axis = 0)

        fkeys = self.__get_final(initial_value)


        if not np.isnan(fkeys[self.out_keys.index("age")]):
            app.T[0] = (app.T[0] / app.T[0].max()) * fkeys[self.out_keys.index("age")]


        for key, value in zip(self.in_keys, initial_value):
            if key in self.out_keys:
                app[0][self.out_keys.index(key)] = value

        if self.max_interp is not None:
            y_max = self.max_interp.test_interpolator(np.array([initial_value]))[0]
            # rescaling y using approx max
            app.T[1:] = ((app.T[1:].T / app.max(axis = 0)[1:]) * y_max).T

        ret = [app] if get_neighbors == False else [app], (k_nearest, distances)
        if get_neighbors == False:
            ret = [app]
        else:
            ret = [app], (k_nearest, distances)         


        return ret

    def __test_average_knn(self, initial_value, class_label):

        k_nearest, distances = self.__k_nearest(initial_value, 4, class_label)

        weights = 1 / distances

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
                tb = [age]
                
                if len(self.out_keys) > 2:
                    tb.extend(interp(age))
                else:
                    tb.append(interp(age))

                temp.append(tb)

                age += dt

            avg.append(temp)

        avg = np.average(np.array(avg), axis = 0)#, weights = weights)

        return [avg]

    def __find_changepoints(self, track):

        # normalizing dependent variables
        track_copy = np.copy(track)

        track = ((track.T - track.min(axis = 0)[:, None]) / (track.max(axis = 0) - track.min(axis = 0) + 1.0e-8)[:, None]).T

        subset = [track.T[1:].T[0], track.T[1:].T[-1]]
        age = [track.T[0][0], track.T[0][-1]]
        inds = [0, track.shape[0] - 1]

        for i in range(self.n_points):
            order = np.array(age).argsort()
            interp = interp1d(np.array(age)[order], np.array(subset)[order].T)

            max_err = -np.inf
            age_to_add = age[0]
            sub_to_add = subset[0]
            ind_to_add = 0

            for ind, tb in enumerate(track):

                err = np.abs(tb[1:] - interp(tb[0])).mean()
                if err > max_err:
                    max_err = err
                    age_to_add =  tb[0]
                    sub_to_add = tb[1:]
                    ind_to_add = ind

            age.append(age_to_add)
            subset.append(sub_to_add)
            inds.append(ind_to_add)

        # ret = np.vstack([age, np.array(subset).T])
        # order = ret[0].argsort()
        inds = np.array(inds)
        inds.sort()

        return track_copy[inds], inds


    def __k_nearest(self, initial_value, k, class_label):

        class_inds = np.where(self.classes == class_label)[0]

        initial_value = (initial_value - self.iv_min) / (self.iv_max - self.iv_min)

        distances = np.sqrt(np.sum(np.square(np.log10(self.iv[class_inds]) - np.log10(initial_value)), axis = 1))
        order = distances.argsort()[:k]

        return self.tracks[class_inds[order]], distances[order]

    def __get_final(self, iv):
        return self.classifier.test_interpolator(np.array([iv]))[0]
    