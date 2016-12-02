import matplotlib.pylab as plt
import numpy as np
from numpy.lib.recfunctions import merge_arrays, append_fields
import os
import cv2
from synchronization import MultipleVideoSynchronization
import parameters
import logging
import pickle
import datetime
import imagesource
from imagesourcevideo import ImageSourceVideo
from imagesource import ImageSource
import subprocess
from joblib import Memory
import sys
import itertools
from sklearn import linear_model
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
memory = Memory(cachedir='.', verbose=0)


class ImageSourceTimedVideo(ImageSourceVideo):
    def __init__(self, filename, mask=None):
        super(ImageSourceTimedVideo, self).__init__(filename, mask)
        self.timestamps_ms = None  # e.g. array([    0.,    40.,    80.,   120., ...])
        self.__extract_timestamps__ = memory.cache(self.__extract_timestamps__)

    def extract_timestamps(self, duration_s=None):
        self.timestamps_ms = self.__extract_timestamps__(duration_s)

    def __extract_timestamps__(self, duration_s):
        # throws CalledProcessError
        ffprobe_cmd = 'ffprobe -select_streams v -show_frames -show_entries ' \
                      'frame=best_effort_timestamp_time %s -of csv' % self.filename
        if duration_s:
            ffprobe_cmd += ' -read_intervals %%%d' % duration_s
        ffprobe_csv = subprocess.check_output(ffprobe_cmd.split()).decode('utf8')
        ffprobe_timestamps = np.recfromcsv(ffprobe_csv.split('\n'), usecols=1,
                                           names='best_effort_timestamp_time')
        return ffprobe_timestamps['best_effort_timestamp_time'] * 1000.

    def get_frame_for_time(self, time_ms):
        assert self.timestamps_ms is not None
        idx = np.searchsorted(self.timestamps_ms, time_ms)
        assert abs(self.timestamps_ms[idx] - time_ms) < 1000  # out of timestamps_ms range
        if idx == 0:
            return idx
        if abs(self.timestamps_ms[idx] - time_ms) > abs(self.timestamps_ms[idx - 1] - time_ms):
            return idx
        else:
            return idx - 1


# class ImageSourceWithFlashEvents(ImageSource):
#     def __init__(self, image_source):
#         self.image_source = image_source
#
#     def get_image(self, frame):
#         return self.image_source.get_image(frame)
#
#     def get_next_image(self):
#         return self.image_source.get_next_image()
#
#     def rewind(self):
#         self.image_source.rewind()
#
#     def

def compute_luminance_median(img):
    return np.median(cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 0], axis=1)

@memory.cache
def extract_features(filename, feature_func, frame_start=0, frame_end=-1, dtype=np.float16):
    # features = np.zeros((image_sequence.get_image(0, camera).shape[0], frame_end - frame_start), dtype=dtype)
    image_source = ImageSourceVideo(filename)
    features = []
    if frame_end == -1:
        frame_end = int(image_source.frame_count)
    image_source.seek(frame_start)
    for i, frame in enumerate(xrange(frame_start, frame_end)):
        try:
            img = image_source.get_next_image()
        except IOError:
            break
        features.append(feature_func(img))
        if (i % 10) == 0:
            logging.info("%d / %d" % (i, frame_end - frame_start))
    return np.array(features, dtype=dtype).T

def ramp_detection(profile, ramp_detection_thresh=4):
    max_pos = np.argmax(profile)
    try:
        start = np.flatnonzero(profile[:max_pos] > ramp_detection_thresh)[0]
        end = max_pos + np.flatnonzero(profile[max_pos:] > ramp_detection_thresh)[-1]
        middle = (start + end) / 2.
    except IndexError:
        start = np.nan
        end = np.nan
        middle = np.nan
    return middle, start, end


def detect_events(features2d, timestamps, hidden_scanlines=0, diff_max_peak_thresh=20, ramp_detection_thresh=4):
    diff = np.diff(features2d.astype(float), axis=1)
    n_scanlines = features2d.shape[0] + hidden_scanlines
    diff_max = np.max(diff, axis=0)
    idx = np.nonzero(diff_max > diff_max_peak_thresh)[0] + 1
    events = []
    for frame in idx:
        profile = diff[:, frame - 1]
        middle, start, end = ramp_detection(profile, ramp_detection_thresh)
        if np.isnan(start):
            logging.warning('unable to detect event ramp in frame %d' % frame)
        else:
            events.append((frame, start, end, timestamps[frame]))
    return np.array(events, dtype=[('frame', int), ('start', float), ('end', float), ('timing', float)])


@memory.cache
def detect_events_in_video(filename, config=None):
    if not config:
        config = {'hidden_scanlines': 0,
                  'diff_max_peak_thresh': 20,
                  'ramp_detection_thresh': 4,
                  }
    features = extract_features(filename, compute_luminance_median, dtype=np.uint8)
    source = ImageSourceTimedVideo(filename)
    source.extract_timestamps()
    events = detect_events(features, source.timestamps_ms,
                  config['hidden_scanlines'],
                  config['diff_max_peak_thresh'],
                  config['ramp_detection_thresh'])
    return events


class FlashSynchronization(object):
    def __init__(self):
        self.events = {}
        self.DIFFMAX_PEAK_THRESH = 20
        self.RAMP_DETECTION_DIFF = 4
        self.MATCH_EVENTS_CLOSENESS_MS = 80 # 35

    def show_events(self, offsets=None):
        assert len(self.events)
        cameras = list(self.events.keys())
        if offsets is None:
            offsets = {cam: 0 for cam in cameras}
        event_timings = {cam: self.events[cam]['timing'] - offsets[cam]
                         for cam in cameras}
        plt.eventplot(event_timings.values(), linelengths=0.95)
        plt.yticks(np.arange(len(cameras)), cameras)
        plt.ylabel('cameras')
        max_timing = max([max(t) for t in event_timings.values()])
        # plt.xticks(np.arange(0, max_timing, step=200000),
        #            (np.arange(0, max_timing, step=200000) / 1000).astype(int))
        plt.xlabel('time in seconds')

    def detect_flash_events(self, filenames):
        self.events = {cam: detect_events_in_video(filename) for cam, filename in filenames.iteritems()}

    def __match_events__(self, offsets, base_cam):
        assert self.events
        nan_val = np.nan
        cameras = offsets.keys()
        timings_rel = {cam: self.events[cam]['timing'] - offsets[cam] for cam in cameras}
        n_timings = {cam: len(self.events[cam]) for cam in cameras}
        cam2idx = {cam: i for i, cam in enumerate(cameras)}
        for cam in cameras:
            assert np.all(np.diff(self.events[cam]['timing']) > 0)
        position = {cam: 0 for cam in cameras}
        all_matched = []
        while position[base_cam] < n_timings[base_cam]:
            matched = [nan_val] * len(cameras)
            matched[cam2idx[base_cam]] = position[base_cam]
            position[base_cam] += 1
            for cam in set(cameras) - {base_cam}:
                while position[cam] < n_timings[cam] and \
                                timings_rel[cam][position[cam]] < timings_rel[base_cam][
                            matched[cam2idx[base_cam]]] - self.MATCH_EVENTS_CLOSENESS_MS:
                    single_event = [nan_val] * len(cameras)
                    single_event[cam2idx[cam]] = position[cam]
                    all_matched.append(single_event)
                    position[cam] += 1
                if position[cam] != n_timings[cam] and \
                                timings_rel[cam][position[cam]] < timings_rel[base_cam][
                            matched[cam2idx[base_cam]]] + self.MATCH_EVENTS_CLOSENESS_MS:
                    matched[cam2idx[cam]] = position[cam]
                    position[cam] += 1
            all_matched.append(matched)

        for cam in set(cameras) - {base_cam}:
            while position[cam] < n_timings[cam]:
                single_event = [nan_val] * len(cameras)
                single_event[cam2idx[cam]] = position[cam]
                all_matched.append(single_event)
                position[cam] += 1

        return np.array(all_matched)

    def get_matched_events(self, offsets, base_cam):
        assert self.events
        matched = self.__match_events__(offsets, base_cam)
        mask_full_match = ~np.isnan(matched.sum(axis=1))
        matched_events = {}
        for idx, cam in enumerate(offsets.keys()):
            matched_events[cam] = self.events[cam][matched[mask_full_match].astype(int)[:, idx]]
        return matched_events

    def fit_synchronization_model(self, offsets, base_cam=None, parameters=None):
        # offsets e.g. {1: 82302.0, 3: 103713.0}
        # parameters e.g. {1: {'sensor_rows': 2625, 'mode_duration': 40}, ... }
        assert len(offsets) == 2
        cam1, cam2 = offsets.keys()
        if base_cam is None:
            base_cam = min(cam1, cam2)
        events_ij = self.get_matched_events(offsets, base_cam)
        if parameters:
            # alpha * t_f + beta + r / (R * fps)
            assert cam1 in parameters and cam2 in parameters
            p = parameters
            X = np.vstack((
                events_ij[cam2]['timing'],
            )).T
            y = events_ij[cam1]['timing'] + \
                events_ij[cam1]['start'] / p[cam1]['sensor_rows'] * p[cam1]['mode_duration_ms'] \
                - events_ij[cam2]['start'] / p[cam2]['sensor_rows'] * p[cam2]['mode_duration_ms']
        else:
            X = np.vstack((
                events_ij[cam2]['timing'],
                events_ij[cam2]['start'],
                -events_ij[cam1]['start'],
            )).T
            y = events_ij[cam1]['timing']

        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac.fit(X, y)

        shift = model_ransac.estimator_.intercept_[0]
        skew = model_ransac.estimator_.coef_[0][0]
        if parameters:
            cam2_time_per_row = None
            cam1_time_per_row = None
        else:
            cam2_time_per_row = model_ransac.estimator_.coef_[0][1]
            cam1_time_per_row = model_ransac.estimator_.coef_[0][2]

        return [skew, shift, cam1_time_per_row, cam2_time_per_row], np.count_nonzero(~model_ransac.inlier_mask_)

    def get_timing(self, timing1, timing2, start1, start2, coefficients, parameters=None):
        [skew, shift, cam1_time_per_row, cam2_time_per_row] = coefficients
        if parameters is not None:
            p = parameters
            cam1_time_per_row = float(p[0]['mode_duration_ms']) / p[0]['sensor_rows']
            cam2_time_per_row = float(p[1]['mode_duration_ms']) / p[1]['sensor_rows']

        cam1_timing = timing1 + start1 * cam1_time_per_row
        cam2_timing = timing2 * skew + shift + start2 * cam2_time_per_row
        return cam1_timing, cam2_timing

    # def synchronize(self, offsets):
    #     cameras = offsets.keys()
    #     t = []
    #
    #     i = None
    #     j = None
    #
    #     for cam1, cam2 in itertools.combinations(cameras, 2):
    #



    def filter_events(self, obsolete_regions={}, override_good={}, override_bad={}, override_start={}):
        # filter out wrong detections
        events = {}
        for cam in selected_cameras:
            cam_events = self.events[cam]
            mask_not_splitted_events = (cam_events['start'] > obsolete_regions[cam]['top']) & \
                               (cam_events['end'] < obsolete_regions[cam]['bottom'])
            event_length_px = np.median(cam_events[mask_not_splitted_events]['end'] -
                                        cam_events[mask_not_splitted_events]['start'])
            if not override_bad or cam not in override_bad or not override_bad[cam]:
                override_bad_mask = np.zeros(len(cam_events), dtype=bool)
            else:
                override_bad_mask = np.any([cam_events['frame'] == frame for frame in override_bad[cam]], axis=0)
            mask_bad = (~(cam_events['end'] == obsolete_regions[cam]['bottom']) &
                        ((cam_events['end'] - cam_events['start']) < event_length_px * 0.9)) | \
                        ((cam_events['end'] - cam_events['start']) > event_length_px * 1.1) | \
                        (cam_events['start'] <= obsolete_regions[cam]['top']) | \
                       override_bad_mask

            if not override_good or cam not in override_good or not override_good[cam]:
                override_good_mask = np.zeros(len(cam_events), dtype=bool)
            else:
                override_good_mask = np.any([cam_events['frame'] == frame for frame in override_good[cam]], axis=0)
            events[cam] = cam_events[~mask_bad | override_good_mask]
            if override_start and cam in override_start:
                for frame, start in override_start[cam]:
                    events[cam][np.where(events[cam]['frame'] == frame)[0][0]]['start'] = start
        self.events = events


class UndistortedImageSource(imagesource.ImageSource):
    def __init__(self, image_source, calibrated_camera):
        self.image_source = image_source
        self.calibrated_camera = calibrated_camera

    def get_image(self, frame):
        img = self.image_source.get_image(frame)
        return self.calibrated_camera.undistort_image(img)

    def get_next_image(self):
        img = self.image_source.get_next_image()
        return self.calibrated_camera.undistort_image(img)

    def rewind(self):
        self.image_source.rewind()


if __name__ == '__main__x':
    sequence_length_sec = 600
    ocred_timings = 'video/usa_rus/frame_timings.pkl'
    root = '../data/ihwc2015/'
        # '../data/ihwc2015/video/usa_rus/'

    out_dir = 'out/'
    features_file = os.path.join(out_dir, 'flashes2d_luminance_median_noseek.pkl')
    out_feature_images = os.path.join(out_dir, '%d.png')

    # /home/matej/prace/sport_tracking/git/experiments/2016-08-22_subframe_synchronization
    p = parameters.Parameters('parameters.yaml')
    # p.c['data_root'] = root
    del p.c['background_subtraction']['masks']
    # bgs = p.get_foreground_sequence()
    images = p.get_image_sequence()
    sync = MultipleVideoSynchronization()
    sync.load(os.path.join(p.c['data_root'], ocred_timings))
    match_start = np.datetime64('1900-01-01T' + p.c['match_start'])

    if os.path.exists(features_file):
        with open(features_file, 'rb') as fr:
            features = pickle.load(fr)
            features_start = pickle.load(fr)
    else:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        features = {}
        features_start = {}
        print datetime.datetime.now()

        for cam in p.c['cameras']:
            start = np.searchsorted(sync.get_timings()[cam], match_start)
            end = np.searchsorted(sync.get_timings()[cam], match_start + np.timedelta64(sequence_length_sec, 's'))
            features_start[cam] = start
            features[cam] = extract_features(images, compute_luminance_median, cam, start, end, dtype=np.uint8)
            print datetime.datetime.now()

            # img = cv2.normalize(features[cam].astype(float),
            #                     np.zeros_like(features[cam], dtype=float),
            #                     0, 255, cv2.NORM_MINMAX, dtype=8)
            # cv2.imwrite(out_feature_images % cam, img)

        with open(features_file, 'wb') as fw:
            pickle.dump(features, fw)
            pickle.dump(features_start, fw)

if __name__ == '__main__':
    selected_cameras = [1, 2, 3, 4]
    filenames = {cam: '../data/ihwc2015/video/usa_rus/%d.mp4' % cam for cam in selected_cameras}

    # selected_cameras = [1, 3]
    # filenames = {cam: '/home/matej/nobackup/sporttracking/subframe_synchronization/%d.mp4' % cam
    #              for cam in selected_cameras}

    sources = {cam: ImageSourceTimedVideo(filenames[cam])
              for cam in selected_cameras}
    sync = FlashSynchronization()

    if len(filenames) == 2:
        sync.detect_flash_events(filenames)

        matching_events = {1: 3, 3: 2, 2: 8, 4: 2}
        offsets = {cam: sync.events[cam][matching_events[cam]]['timing'] for cam in selected_cameras}

        images_dimensions = {}
        for cam in selected_cameras:
            img = sources[cam].get_image(0)
            images_dimensions[cam] = img.shape
        obsolete_regions = {cam: {'top': 28, 'bottom': images_dimensions[cam][0] - 1}
                            for cam in selected_cameras}
        override_good = {0: [], 1: [1917, 5983, 10718], 2: [3605, 5122, 14935], 3: [6415, ], 4: []}  # force to stay
        override_start = {0: [], 1: [(10718, 983), ], 2: [], 3: [(6415, 633), ], 4: []}  # force to stay and fix start
        override_bad = {0: [], 1: [], 2: [12992, ], 3: [], 4: []}  # force to filter out
        sync.filter_events(obsolete_regions, override_good, override_bad, override_start)

        # with open('flashsynchronization_4cam_full.pkl', 'wb') as fw:
        #     pickle.dump(sync.__features__, fw)
        #     pickle.dump(sync.events, fw)
        #     pickle.dump(offsets, fw)
    else:
        with open('flashsynchronization_4cam_full.pkl', 'rb') as fr:
            sync.__features__ = pickle.load(fr)
            sync.events = pickle.load(fr)
            # offsets = pickle.load(fr)

        offsets = {1: 65177.000000000007, 2: 59096.0, 3: 102673.0, 4: 89037.0}

        # find events (state before sync.filter_events)
        for cam in selected_cameras:
            sources[cam].extract_timestamps()
        sync.events = {cam: detect_events(sync.__features__[cam], sources[cam].timestamps_ms) for cam in selected_cameras}

        # 10 minutes from the game start
        visapp_used_frames = {1: (1669, 16564), 2: (1517, 16412), 3: (2203, 14723), 4: (1926, 14590)}
        for cam in selected_cameras:
            visapp_mask = (sync.events[cam]['frame'] > visapp_used_frames[cam][0]) & \
                          (sync.events[cam]['frame'] < visapp_used_frames[cam][1])
            sync.events[cam] = sync.events[cam][visapp_mask]

        print {cam: len(sync.events[cam]) for cam in selected_cameras}

        images_dimensions = {}
        for cam in selected_cameras:
            img = sources[cam].get_image(0)
            images_dimensions[cam] = img.shape
        obsolete_regions = {cam: {'top': 28, 'bottom': images_dimensions[cam][0] - 1}
                            for cam in selected_cameras}
        override_good = {0: [], 1: [1917, 5983, 10718], 2: [3605, 5122, 14935], 3: [6415, ], 4: []}  # force to stay
        override_start = {0: [], 1: [(10718, 983), ], 2: [], 3: [(6415, 633), ], 4: []}  # force to stay and fix start
        override_bad = {0: [], 1: [], 2: [12992, ], 3: [], 4: []}  # force to filter out
        sync.filter_events(obsolete_regions, override_good, override_bad, override_start)


    print {cam: len(sync.events[cam]) for cam in selected_cameras}
    # sync.show_events(offsets)

    # matched = sync.__match_events__({cam: sync.events[cam]['timing'] for cam in selected_cameras},
    #                       selected_cameras, offsets, 1)
    #
    # mask_full_match = ~np.isnan(matched.sum(axis=1))


    matching_events = {1: 0, 3: 0, 2: 0, 4: 0}
    offsets = {cam: sync.events[cam][matching_events[cam]]['timing'] for cam in selected_cameras}


    matched_events = sync.get_matched_events(offsets, 1)

    parameters = {0: {'sensor_rows': -1,   'mode_duration_ms': -1},
                  1: {'sensor_rows': 2625, 'mode_duration_ms': 40},
                  2: {'sensor_rows': 2625, 'mode_duration_ms': 40},
                  3: {'sensor_rows': 978,  'mode_duration_ms': 40},
                  4: {'sensor_rows': 978,  'mode_duration_ms': 40},
                  }

    # def synchronize(self, offsets):
    #     cameras = offsets.keys()
    #     t = []
    #
    #     i = None
    #     j = None
    #
    t = []
    for cam1, cam2 in itertools.combinations(offsets.keys(), 2):
        events_ij = sync.get_matched_events({cam: offsets[cam] for cam in [cam1, cam2]}, base_cam=min([cam1, cam2]))

        # coeffs, outliers = sync.fit_synchronization_model({cam: offsets[cam] for cam in [cam1, cam2]},
        #                                base_cam=min([cam1, cam2]), parameters=parameters)
        # timing1, timing2 = sync.get_timing(events_ij[cam1]['timing'], events_ij[cam2]['timing'],
        #                                    events_ij[cam1]['start'], events_ij[cam2]['start'],
        #                                    coeffs, [parameters[cam] for cam in [cam1, cam2]])


        coeffs, outliers_ = sync.fit_synchronization_model({cam: offsets[cam] for cam in [cam1, cam2]},
                                                            base_cam=min([cam1, cam2]))
        timing1, timing2 = sync.get_timing(events_ij[cam1]['timing'], events_ij[cam2]['timing'],
                                           events_ij[cam1]['start'], events_ij[cam2]['start'],
                                           coeffs)
        print cam1, cam2, (timing1 - timing2).std()

        n = len(events_ij[cam1])
        [skew, shift, cam1_time_per_row, cam2_time_per_row] = coeffs
        # clock2_skew_lines_per_second = (alpha - 1) * sensor_rows[cam2] / mode_duration[cam2] * 1000
        t.append(['%d %d' % (cam1, cam2),
                  n,
                  skew - 1,
                  shift,
                  cam1_time_per_row,
                  cam2_time_per_row,
                  (timing1 - timing2).std(),
                  # clock2_skew_lines_per_second,
                  # np.count_nonzero(~model_ransac.inlier_mask_),
                  ])
        headers = ['camera', 'number of events', 'skew', 'shift (in ms)', 't/row cam1', 't/row cam2', 'std (in ms)'] # , 'outliers'] # 'clock skew',

    print(tabulate(t, headers=headers))

# prebyva:
#
# sync.events[4][10]
# Out[6]:
# (5536, 28.0, 119.0, 5536.038888888889, 261671.0)
