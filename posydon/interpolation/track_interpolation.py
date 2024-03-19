""" Module for performing track interpolation """

__authors__ = [
    "Philipp Moura Srivastava <philipp.msrivastava@northwestern.edu>"
]

import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import torch

from posydon.interpolation.IF_interpolation import IFInterpolator
from posydon.utils.common_functions import PATH_TO_POSYDON_DATA
from posydon.grids.downsampling import TrackDownsampler
from posydon.interpolation.utils import set_valid
from posydon.grids.psygrid import PSyGrid
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
from itertools import compress
import numpy as np
from tqdm import tqdm
import os

def find_changepoints(track, n_points):

    # normalizing dependent variables
    track_copy = np.copy(track)

    track = ((track.T - track.min(axis = 0)[:, None]) / (track.max(axis = 0) - track.min(axis = 0) + 1.0e-8)[:, None]).T

    subset = [track.T[1:].T[0], track.T[1:].T[-1]]
    age = [track.T[0][0], track.T[0][-1]]
    inds = [0, track.shape[0] - 1]

    for i in range(n_points):
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

    inds = np.array(inds)
    inds.sort()

    return track_copy[inds], inds


def k_nearest(self, initial_value, k, class_label):

    class_inds = np.where(self.classes == class_label)[0]

    initial_value = (np.log10(initial_value + 1.0e-8) - self.iv_min) / (self.iv_max - self.iv_min)

    distances = np.sqrt(np.sum(np.square(self.iv[class_inds] - initial_value), axis = 1))

    order = distances.argsort()[:k]

    return self.tracks[class_inds[order]], distances[order]

def create_dataset(grid, klass, n_points, n_tracks, in_keys, out_keys, save_path, interp_in_q = False):

    iv = np.array(grid.initial_values[in_keys].tolist())
    classes = grid.final_values["interpolation_class"]
    unique_classes = np.unique(classes)

    valid_inds = set_valid(classes, iv, unique_classes,
                interp_in_q)

    # taking out systems that start in RLOF
    valid_inds[np.where((grid.final_values["termination_flag_1"] == "forced_initial_RLO"))[0]] = -1
    valid_inds[np.where((grid.final_values["interpolation_class"] != klass))[0]] = -1
    valid_bool = lambda track: track.history1 is None or track.history2 is None or track.binary_history is None if interp_in_q else track.history1 is None or track.binary_history is None
    valid_inds[[ind for ind, track in enumerate(grid) if valid_bool(track)]] = -1

    # normalizing inputs
    niv = (iv - iv.min(axis = 0)) / (iv.max(axis = 0) - iv.min(axis = 0))
    niv = niv[valid_inds >= 0]

    # getting preprocessed tracks
    tracks = []

    bhist_keys = [k for k in out_keys if k.split("_")[0] != "S1" and k.split("_")[0] != "S2"]
    s1_keys = [k[3:] for k in out_keys if k.split("_")[0] == "S1"]
    s2_keys = [k[3:] for k in out_keys if k.split("_")[0] == "S2"]

    for ind, v in enumerate(valid_inds):
        if v >= 0:
            if interp_in_q:
                bin_hist = np.array(grid[ind].binary_history[bhist_keys].tolist())
                s1_hist = np.array(grid[ind].history1[s1_keys].tolist())
                s2_hist = np.array(grid[ind].history2[s2_keys].tolist())

                track = np.hstack([bin_hist, s1_hist, s2_hist])

            else:
                bin_hist = np.array(grid[ind].binary_history[bhist_keys].tolist())
                s1_hist = np.array(grid[ind].history1[s1_keys].tolist())

                track = np.hstack([bin_hist, s1_hist])

            cp_track, _ = find_changepoints(track, n_points)

            tracks.append(cp_track)
    
    tracks = np.array(tracks)

    # normalizing tracks
    t_min = tracks.min(axis = (0, 1))
    t_max = tracks.max(axis = (0, 1))

    tracks = (tracks - t_min) / (t_max - t_min + 1.0e-8)

    features = []
    nearest_neighbors = []
    gt = []

    for ind, initial_value in enumerate(niv):

        gt.append(tracks[ind])
        # getting nearest neighbors
        nearest = np.sqrt(np.square(initial_value - niv).sum(axis = 1)).argsort()[1:n_tracks + 1]
        features.append(np.vstack([initial_value, niv[nearest]]))
        nearest_neighbors.append(tracks[nearest])

    np.savez(save_path, features = features, nearest_neighbors = nearest_neighbors, gt = gt)



class Stars(Dataset):

    def __init__(self, feats, neighs, gts, inds = None):

        self.b_ini = feats
        self.feats = np.array([np.array(n).T[inds] for n in neighs])
        self.neighs = np.array([np.array(n).T[inds] for n in neighs])
        self.gts = np.array([np.array(gt).T[inds] for gt in gts])

    def __len__(self):

        return self.feats.shape[0]

    def __getitem__(self, idx):

        b_ini = torch.from_numpy(np.array(self.b_ini[idx], dtype = np.float32))
        b_ini = b_ini.flatten()[:3]

        feat = torch.from_numpy(np.hstack(np.array(self.feats[idx], dtype = np.float32)))
        neigh = torch.from_numpy(np.array(self.neighs[idx], dtype = np.float32))
        gt = torch.from_numpy(np.array(self.gts[idx], dtype = np.float32))

        return b_ini, feat, neigh, gt 


class Encoder(nn.Module):

    def __init__(self, dim_in, dim_hid, dim_out):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.ln1 = nn.LayerNorm(dim_hid)

        self.fc2 = nn.Linear(dim_hid, dim_hid)
        self.ln2 = nn.LayerNorm(dim_hid)

        self.fc3 = nn.Linear(dim_hid, dim_out)

    def forward(self, x):

        output = F.relu(self.ln1(self.fc1(x)))
        output = F.relu(self.ln2(self.fc2(output)))
        output = self.fc3(output)

        return output



class MLP(nn.Module):

    def __init__(self, pos_dim, dim_in, dim_hid, dim_out):
        super(MLP, self).__init__()

        self.encoder = Encoder(pos_dim, 8, 8)

        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3, padding = "same")
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, padding = "same")
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, padding = "same")

        self.fc1 = nn.Linear(40, dim_hid)
        self.ln1 = nn.LayerNorm(dim_hid)

        self.fc2 = nn.Linear(dim_hid, dim_hid)
        self.ln2 = nn.LayerNorm(dim_hid)

        self.fc3 = nn.Linear(dim_hid, dim_hid)
        self.ln3 = nn.LayerNorm(dim_hid)

        self.fc4 = nn.Linear(dim_hid, dim_hid)
        self.ln4 = nn.LayerNorm(dim_hid)

        self.fc5 = nn.Linear(dim_hid, dim_hid)
        self.ln5 = nn.LayerNorm(dim_hid)

        self.fc6 = nn.Linear(dim_hid, dim_hid)
        self.ln6 = nn.LayerNorm(dim_hid)

        self.fc7 = nn.Linear(dim_hid, dim_out)

    def forward(self, b_i, neighbors):

        # pos_b_i = self.encoder(b_i)
        # pos_b_i = pos_b_i.reshape(neighbors.shape) # reshaping to match track image

        # # adding positions
        # pos_neighbors = neighbors + pos_b_i
        
        lat_b_ini = self.encoder(b_i)
        
        # addng channel
        # pos_neighbors = neighbors.unsqueeze(1)

        # # extracting features
        # output = F.max_pool2d(F.relu(self.conv1(pos_neighbors)), kernel_size = 2, stride = 1)
        # output = F.max_pool2d(F.relu(self.conv2(output)), kernel_size = 2, stride = 1)
        # output = F.max_pool2d(F.relu(self.conv3(output)), kernel_size = 2, stride = 1)

        # flattening
        output = neighbors.flatten(start_dim = 1)
        output = torch.hstack([output, lat_b_ini])

        output = F.relu(self.ln1(self.fc1(output)))
        output = F.relu(self.ln2(self.fc2(output)))
        output = F.relu(self.ln3(self.fc3(output)))
        output = F.relu(self.ln4(self.fc4(output)))
        output = F.relu(self.ln5(self.fc5(output)))
        output = F.relu(self.ln6(self.fc6(output)))

        output = self.fc7(output)

        return output


class Simple(nn.Module):

    def __init__(self, dim_in, hid_dim, out_dim, pos_dim):
        super(Simple, self).__init__()

        self.track_enc = Encoder(dim_in, hid_dim, 32)
        self.init_enc = Encoder(pos_dim, hid_dim, 32)

        self.fc1 = nn.Linear(64, hid_dim)
        self.ln1 = nn.LayerNorm(hid_dim)

        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)

        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.ln3 = nn.LayerNorm(hid_dim)

        self.fc4 = nn.Linear(hid_dim, hid_dim)
        self.ln4 = nn.LayerNorm(hid_dim)

        self.fc5 = nn.Linear(hid_dim, out_dim)

    def forward(self, b_init, track):

        lb_init = self.init_enc(b_init)
        ltrack = self.track_enc(track)

        pos_track = torch.hstack([lb_init, ltrack])

        output = F.relu(self.ln1(self.fc1(pos_track)))
        output = F.relu(self.ln2(self.fc2(output)))
        output = F.relu(self.ln3(self.fc3(output)))
        output = F.relu(self.ln4(self.fc4(output)))

        output = self.fc5(output)

        return output



class TrackInterpolator:

    def __init__(self, in_keys, out_keys, method, phase, n_points = 9, rescale = True):

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
        self.rescale = rescale

    def load(self, tracks, labels, classifier, inds, classes, iv, iv_max, iv_min):
        self.tracks = tracks
        self.labels = labels
        self.classifier = classifier
        self.key_inds = inds

        self.classes = classes
        self.iv = iv
        self.iv_max = iv_max
        self.iv_min = iv_min

    def train(self, grid, networks = None):

        self.networks = networks
        self.tracks = []

        self.iv = np.array(grid.initial_values[self.in_keys].tolist())
        self.classes = grid.final_values["interpolation_class"]
        unique_classes = ["stable_MT", "unstable_MT", "no_MT"]

        valid_inds = set_valid(self.classes, self.iv, unique_classes,
                    self.phase == "HMS-HMS")

        # taking out systems that start in RLOF
        valid_inds[np.where((grid.final_values["termination_flag_1"] == "forced_initial_RLO"))[0]] = -1
        valid_bool = lambda track: track.history1 is None or track.history2 is None or track.binary_history is None if self.phase == "HMS-HMS" else track.history1 is None or track.binary_history is None
        valid_inds[[ind for ind, track in enumerate(grid) if valid_bool(track)]] = -1

        self.iv = self.iv[valid_inds >= 0] # initial values
        self.iv = np.log10(self.iv + 1.0e-8)
        self.iv_max, self.iv_min = self.iv.max(axis = 0), self.iv.min(axis = 0)
        
        self.iv = (self.iv - self.iv_min) / (self.iv_max - self.iv_min)

        self.classes = self.classes[valid_inds >= 0]

        bhist_keys = [k for k in self.out_keys if k.split("_")[0] != "S1" and k.split("_")[0] != "S2"]
        s1_keys = [k[3:] for k in self.out_keys if k.split("_")[0] == "S1"]
        s2_keys = [k[3:] for k in self.out_keys if k.split("_")[0] == "S2"]

        mask = np.where(np.array([len(bhist_keys) > 0, len(s1_keys) > 0, len(s2_keys) > 0]) == True)[0]

        if self.phase != "HMS-HMS" and len(s2_keys) > 0:
            # self.out_keys = [k for k in self.out_keys if k[3:] not in s2_keys] + s1_keys
            print("Star 2 is undefined for grids containing CO")

        for ind, v in tqdm(enumerate(valid_inds)):

            if v >= 0:
                
                if self.phase == "HMS-HMS":
                    bin_hist = np.array(grid[ind].binary_history[bhist_keys].tolist())
                    s1_hist = np.array(grid[ind].history1[s1_keys].tolist())
                    s2_hist = np.array(grid[ind].history2[s2_keys].tolist())

                    to_append = np.hstack([bin_hist, s1_hist, s2_hist])
                else:
                    bin_hist = np.array(grid[ind].binary_history[bhist_keys].tolist())
                    s1_hist = np.array(grid[ind].history1[s1_keys].tolist())

                    to_concat = list(compress([bin_hist, s1_hist], mask))
                    to_append = np.concatenate(to_concat, axis = 1)

                # parameters = []


                # for k in range(1, len(self.out_keys)):
                #     cps, _ = find_changepoints(np.array([to_append.T[0], to_append.T[k]]).T, n_points = self.n_points)
                #     parameters.append(cps)

                # self.tracks.append(np.array(parameters).T)

                # cps, _ = find_changepoints(to_append, n_points = self.n_points)

                self.tracks.append(to_append)


        self.tracks = np.array(self.tracks)

        self.labels = np.unique(self.classes)

        IFInterp = IFInterpolator()

        IFInterp.load(os.path.join(PATH_TO_POSYDON_DATA, "POSYDON_data", self.phase,
                                    "interpolators", "linear3c_kNN", "grid_0.0142.pkl"))

        self.classifier = IFInterp.interpolators[0]
        self.key_inds = np.array([self.classifier.out_keys.index(key) for key in self.out_keys])

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
                meta_data.append(m[0])

            else:
                approx.extend(self.method(iv, _class))

        return np.array(approx), np.array(meta_data) if meta else np.array(approx)

    def evaluate(self, binary):
        pass

    def save(self):
        pass

    def __test_1nn(self, initial_value, class_label, meta = False):
        nearest = k_nearest(self, initial_value, 1, class_label)[0]

        if self.rescale:
            y_pred = self.classifier.test_interpolator(np.array([initial_value]))[0]
            nearest[0].T[1:] = ((nearest[0].T[1:].T / nearest[0][-1][1:]) * y_pred[self.key_inds[1:]]).T

        if meta:
            nearest = nearest, [nearest] * 4

        return nearest

    def __test_changepoints_knn(self, initial_value, class_label, get_neighbors = False):
        

        _k_nearest, distances = k_nearest(self, initial_value, 8, class_label)
        
        weights = 1 / np.power(distances, 3)
  
        weights = weights / weights.sum()

        parameters = []
        ages = []
        
        for nearest in _k_nearest:
            ages.append(nearest[0])
            parameters.append(nearest[1])

        neighbors = _k_nearest
        ages = np.array(ages)
        _k_nearest = parameters

        # cps = []

        # for nearest in _k_nearest:
        #     cps.append(find_changepoints(nearest, self.n_points)[0])

        # app = np.average(np.array(cps), axis = 0, weights = weights)

        age_app = np.average(ages, axis = 0, weights = weights)

        # print(np.array(ages)[:, :, 1][:, 12])
        app = np.average(_k_nearest, axis = 0, weights = weights)
        # print(age_app[:, 1][12])

        fkeys = self.__get_final(initial_value)

        # Rescaling age (assuming that age is first key)
        # if not np.isnan(fkeys[self.key_inds[0]]):
        #     age_app = (age_app / (age_app.max(axis = 0) + 1.0e-8)) * fkeys[self.key_inds[0]]
        # print(age_app[:, 1][12])

        if self.rescale:
            y_pred = self.classifier.test_interpolator(np.array([initial_value]))[0]
            # rescaling y using approx max
            app.T[1:] = ((app.T[1:].T / app[-1][1:]) * y_pred[self.key_inds[1:]]).T

        if np.isnan(age_app).any() == True or np.isnan(app).any():
            print(np.isnan(age_app).any(), np.isnan(app).any())
        app = [age_app, app]

        ret = [app] if get_neighbors == False else [app], (_k_nearest, distances)
        if get_neighbors == False:
            ret = [app]
        else:
            ret = [app], (neighbors, weights)

        return ret

    def __test_average_knn(self, initial_value, class_label):

        _k_nearest, distances = k_nearest(self, initial_value, 4, class_label)

        weights = 1 / distances

        avg = []

        for nearest in _k_nearest:

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

    def __get_final(self, iv):
        return self.classifier.test_interpolator(np.array([iv]))[0]
    