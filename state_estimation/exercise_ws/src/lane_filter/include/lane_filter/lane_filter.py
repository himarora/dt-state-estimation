from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt, pi


class LaneFilterHistogramKF():
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'range_min',
            'range_est',
            'range_max',
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])

        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]

        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}
        self.process_out = self.belief["mean"]
        self.measurement_out = self.belief["mean"]
        self.ekf_out = self.belief["mean"]
        self.encoder_resolution = 0
        self.wheel_radius = 0.0
        self.initialized = False
        self.ds = [0.]
        self.phis = [0.]

        # self.Q = np.array([[0.1, 0.01], [0.01,
        #                                  0.3]])  # Works for 1 turn, bot goes into wrong lane, corrects and then takes a u-turn on the second turn
        # self.R = np.array([[0.5, 0.01], [0.01, 0.1]])

        self.Q = np.array([[0.12, 0.01], [0.01, 0.12]])
        self.R = np.array([[0.1, 0.001], [0.001, 0.1]])
        print(f"Q:{self.Q}")
        print(f"R:{self.R}")
        self.tj = np.array([0., 0.])

    @staticmethod
    def f(x_k_1, u_k, dt, r=0.0318, tpr=135, l=0.1):
        c = 2 * pi * r
        vl = (c * u_k[0]) / (tpr * dt)
        vr = (c * u_k[1]) / (tpr * dt)
        v = (vl + vr) * 0.5
        w = (vr - vl) / l
        phi_delta = w * dt
        x_delta = np.array([v * np.sin(phi_delta), phi_delta])
        x_k_ = x_k_1 + x_delta
        return x_k_, v, w

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        if not self.initialized:
            return

        # x
        x_k_1 = self.belief["mean"]
        x_k_, v, w = self.f(x_k_1, [left_encoder_delta, right_encoder_delta], dt)

        # P
        P_k_1 = self.belief["covariance"]
        Q = self.Q
        A_k = np.array([[1, v * dt * np.cos(x_k_[1] - x_k_1[1])], [0, 1]])
        P_k_ = (A_k @ P_k_1 @ A_k.T) + Q
        self.belief = {'mean': x_k_, 'covariance': P_k_}
        self.process_out = x_k_
        # print(f"Pr: {x_k_[0]:.2f} {x_k_[1]:.2f}")

    def update(self, segments):
        # print(f"Lane Filter Update: {self.tj}")
        # prepare the segments for each belief array
        segmentsArray = self.prepareSegments(segments)
        # generate all belief arrays
        # segmentsArray = self.filterSegments(segmentsArray, self.tj)
        measurement_likelihood = self.generate_measurement_likelihood(
            segmentsArray)  # 23 x 30  d x phi      # None if len(segmentsArray = 0)

        # TODO: Parameterize the measurement likelihood as a Gaussian
        if measurement_likelihood is not None:
            d_bin, phi_bin = np.unravel_index(measurement_likelihood.argmax(), measurement_likelihood.shape)
            d, phi = self.d_min + (d_bin * self.delta_d), self.phi_min + (phi_bin * self.delta_phi)
            # print(f"Me: {d:.2f} {phi:.2f}")
            # print(f"d_b:{d_bin:.2f} phi_b:{phi_bin:.2f}")
            # d, phi = get_robust_indices(measurement_likelihood)
            # print(f"d_i:{d:.2f} phi_i:{phi:.2f}")
            self.ds.append(d)
            self.phis.append(phi)
            z_k = np.array([d, phi])
            self.measurement_out = z_k

            # TODO: Apply the update equations for the Kalman Filter to self.belief
            x_k_, P_k_ = self.belief["mean"], self.belief["covariance"]
            H_k = np.array([[1., 0.], [0., 1.]])
            R = self.R
            K_k = P_k_ @ H_k.T @ np.linalg.inv((H_k @ P_k_ @ H_k.T) + R)

            # x
            x_k = x_k_ + K_k @ (z_k - x_k_)

            # P
            I = np.identity(K_k.shape[0])
            P_k = (I - (K_k @ H_k)) @ P_k_

            self.belief = {'mean': x_k, 'covariance': P_k}
            self.ekf_out = x_k
            # print(f"EKF: {x_k[0]:.2f} {x_k[1]:.2f}")

    def getEstimate(self):
        return self.belief

    def generate_measurement_likelihood(self, segments):

        if len(segments) == 0:
            return None

        grid = np.mgrid[self.d_min:self.d_max:self.delta_d,
               self.phi_min:self.phi_max:self.delta_phi]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape)

        for segment in segments:
            d_i, phi_i, l_i, weight = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return None

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / \
                                 np.sum(measurement_likelihood)
        return measurement_likelihood

    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if (p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < 3 * self.delta_d and abs(phi_s - phi_max) < 3 * self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c ** 2 + y_c ** 2)

    def filterSegments(self, segments, eq):
        white_segments = []
        yellow_segments = []
        for i, segment in enumerate(segments):
            to_append = white_segments if segment.color == segment.WHITE else yellow_segments
            to_append.append(i)

        idx = []
        if np.abs(eq[0]) is not None:
            # print(f"Detected white segments: {len(white_segments)}")
            for i in white_segments:
                points = segments[i].points
                point = (
                (points[0].x + points[1].x) * 0.5, (points[0].y + points[1].y) * 0.5)
                if point[1] > 0 and point[0] < 0.2:
                    idx.append(i)

        if np.abs(eq[0]) is not None:
            # print(f"Detected white segments: {len(white_segments)}")
            for i in yellow_segments:
                points = segments[i].points
                point = (
                (points[0].x + points[1].x) * 0.5, (points[0].y + points[1].y) * 0.5)
                if point[1] < 0:
                    idx.append(i)

        if len(idx) > 0:
            new_segments = [seg for i, seg in enumerate(segments) if i not in idx]
            print(f"Filtered {len(idx)} segments")
            segments = new_segments
        return segments

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray


def get_robust_indices(data):
    x1, x2 = data.sum(axis=0), data.sum(axis=1)
    idx = []
    for data in (x1, x2):
        data /= data.sum()
        clusters = []
        clusters_length = []
        clusters_argmax = []
        th = 1e-4
        current_c_i = []
        current_c_v = []
        could_be_out = 0
        for i, v in enumerate(data):
            if v >= th:
                current_c_i.append(i)
                current_c_v.append(v)
            else:
                could_be_out += 1
                if could_be_out > 1:
                    if len(current_c_i):
                        clusters.append(current_c_i)
                        clusters_length.append(len(current_c_i))
                        clusters_argmax.append(np.argmax(current_c_v))
                    current_c_i = []
                    current_c_v = []
                    could_be_out = 0
        if not len(clusters_length):
            idx.append(data.argmax())
        max_cluster = np.argmax(clusters_length)
        idx.append(clusters[max_cluster][clusters_argmax[np.argmax(clusters_length)]])
        idx.reverse()
    return tuple(idx)
