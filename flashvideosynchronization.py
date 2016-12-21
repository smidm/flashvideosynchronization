import os
import logging
import itertools
import math
import matplotlib.pylab as plt
import matplotlib
import numpy as np
import scipy
import cv2
import joblib
from sklearn import linear_model
try:
    import seaborn as sns
    has_seaborn = True
except ImportError:
    has_seaborn = False
import imagesource

logging.basicConfig(level=logging.INFO)
memory = joblib.Memory(cachedir='.', verbose=2)


@memory.cache
def extract_features(filename, frame_start=0, frame_end=-1, dtype=np.float16):
    image_source = imagesource.VideoSource(filename)
    image_source.color_conversion_from_bgr = cv2.COLOR_BGR2Lab
    features = []
    if frame_end == -1:
        if not math.isinf(image_source.frame_count):
            frame_end = int(image_source.frame_count)
            frame_range = xrange(frame_start, frame_end)
        else:
            frame_end = frame_start  # for logging
            frame_range = itertools.count(start=frame_start)
    image_source.seek(frame_start)
    for i, frame in enumerate(frame_range):
        try:
            img = image_source.get_next_image()
        except IOError:
            break
        features.append(np.median(img[:, :, 0], axis=1))
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
    return np.array(events, dtype=[('frame', int), ('start', float), ('end', float), ('time', float)])


@memory.cache
def detect_events_in_video(filename, config=None):
    if not config:
        config = {'hidden_scanlines': 0,
                  'diff_max_peak_thresh': 20,
                  'ramp_detection_thresh': 4,
                  }
    features = extract_features(filename, dtype=np.uint8)
    source = imagesource.TimedVideoSource(filename)
    source.extract_timestamps()
    events = detect_events(features, source.timestamps_ms,
                  config['hidden_scanlines'],
                  config['diff_max_peak_thresh'],
                  config['ramp_detection_thresh'])
    return events


class FlashVideoSynchronization(object):
    def __init__(self):
        self.events = {}
        self.model = {}
        self.base_cam = None
        self.DIFFMAX_PEAK_THRESH = 20
        self.RAMP_DETECTION_DIFF = 4
        self.MATCH_EVENTS_CLOSENESS_MS = 80  # 35

    def detect_flash_events(self, filenames):
        # cashed using joblib.Memory
        self.events = {cam: detect_events_in_video(filename) for cam, filename in filenames.iteritems()}

    def show_events(self, offsets=None):
        assert len(self.events)
        cameras = list(self.events.keys())
        if offsets is None:
            offsets = {cam: 0 for cam in cameras}
        event_times = {cam: self.events[cam]['time'] - offsets[cam]
                         for cam in cameras}
        fig = plt.figure()
        if has_seaborn:
            colors = sns.color_palette(n_colors=len(cameras))
        else:
            colors = None
        plt.eventplot(event_times.values(), colors=colors, linelengths=0.95)
        plt.yticks(np.arange(len(cameras)), cameras)
        plt.ylabel('cameras')
        max_time = max([max(t) for t in event_times.values()])
        plt.xticks(np.arange(0, max_time, step=50000),
                  (np.arange(0, max_time, step=50000) / 1000).astype(int))
        plt.xlabel('time in seconds')
        plt.tight_layout(0)
        if has_seaborn:
            sns.despine(fig)

    def filter_events(self, obsolete_regions={}, override_good={}, override_bad={}, override_start={}):
        # filter out wrong detections
        events = {}
        for cam in self.events.keys():
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
                    idx = np.where(events[cam]['frame'] == frame)[0]
                    if idx:
                        events[cam][idx[0]]['start'] = start
        self.events = events

    def save_event_images(self, sources, features, output_dir, cameras=None):
        if cameras is None:
            cameras = sources.keys()
        fig_width_in = 4
        fig_height_in = fig_width_in * 0.7
        params = {
            'figure.figsize': [fig_width_in, fig_height_in],
            'figure.dpi': 80,
            'savefig.dpi': 150,
            'font.size': 5,
        }
        plt.rcParams.update(params)
        for cam in cameras:
            for e in self.events[cam]:
                fig = plt.figure()
                fig.set_size_inches(fig_width_in * 2, fig_height_in * 2 * 0.6, forward=True)
                self.plot_frame_with_profile(sources[cam].get_image(e['frame']), e['frame'], features[cam],
                                             e['start'], e['end'])
                plt.savefig(os.path.join(output_dir, 'c%s_f%d.jpg' % (str(cam), e['frame'])))
                plt.close(fig)

    def plot_frame_with_profile(self, img, frame_nr, features, start=None, end=None):
        def plot_start_end_lines(start, end):
            if start:
                plt.hlines(start, plt.xlim()[0], plt.xlim()[1], 'r', linestyles='dotted')
            if end:
                plt.hlines(end, plt.xlim()[0], plt.xlim()[1], 'r', linestyles='dotted')

        def set_axes_and_legend(ax):
            ax.spines['left'].set_position('zero')
            plt.locator_params(axis='x', nbins=4)
            ax.axes.yaxis.set_ticks([])
            if has_seaborn:
                sns.despine(ax=ax)
            plt.legend(loc='upper right', fontsize='x-small')

        plt.axis('tight')
        gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[6, 1, 1.5])

        ax1 = plt.subplot(gs[0])
        height = img.shape[0]
        ax1.imshow(img)
        plt.title('frame $I_n$')
        plt.grid(False)
        plt.axis('off')
        plot_start_end_lines(start, end)

        ax = plt.subplot(gs[1], sharey=ax1)
        plt.title('median line\nintensity')
        plt.plot(features[:, frame_nr], range(height), label='$\mathrm{I}_n$')
        plt.plot(features[:, frame_nr - 1], range(height), label='$\mathrm{I}_{n-1}$')
        plot_start_end_lines(start, end)
        set_axes_and_legend(ax)

        ax = plt.subplot(gs[2], sharey=ax1)
        plt.title('median line\nintensity\ndifference')
        plt.plot(features[:, frame_nr].astype(float) - features[:, frame_nr - 1].astype(float),
                 range(height), label='$\mathrm{I}_n - \mathrm{I}_{n-1}$')
        plt.plot(features[:, frame_nr - 1].astype(float) - features[:, frame_nr - 2].astype(float),
                 range(height), label='$\mathrm{I}_{n-1} - \mathrm{I}_{n-2}$')
        plot_start_end_lines(start, end)
        set_axes_and_legend(ax)

        # plt.autoscale(tight=True)
        plt.tight_layout()

    def get_matched_events(self, cameras, offsets, base_cam=None):
        assert self.events
        if base_cam is None:
            base_cam = min(cameras)
        matched = self.__match_events__(cameras, offsets, base_cam)
        mask_full_match = ~np.isnan(matched.sum(axis=1))
        matched_events = {}
        for idx, cam in enumerate(cameras):
            matched_events[cam] = self.events[cam][matched[mask_full_match].astype(int)[:, idx]]
        return matched_events

    def __match_events__(self, cameras, offsets, base_cam):
        assert self.events
        nan_val = np.nan
        time_rel = {cam: self.events[cam]['time'] - offsets[cam] for cam in cameras}
        n_events = {cam: len(self.events[cam]) for cam in cameras}
        cam2idx = {cam: i for i, cam in enumerate(cameras)}
        for cam in cameras:
            assert np.all(np.diff(self.events[cam]['time']) > 0)
        position = {cam: 0 for cam in cameras}
        all_matched = []
        while position[base_cam] < n_events[base_cam]:
            matched = [nan_val] * len(cameras)
            matched[cam2idx[base_cam]] = position[base_cam]
            position[base_cam] += 1
            for cam in set(cameras) - {base_cam}:
                while position[cam] < n_events[cam] and \
                                time_rel[cam][position[cam]] < time_rel[base_cam][
                            matched[cam2idx[base_cam]]] - self.MATCH_EVENTS_CLOSENESS_MS:
                    single_event = [nan_val] * len(cameras)
                    single_event[cam2idx[cam]] = position[cam]
                    all_matched.append(single_event)
                    position[cam] += 1
                if position[cam] != n_events[cam] and \
                                time_rel[cam][position[cam]] < time_rel[base_cam][
                            matched[cam2idx[base_cam]]] + self.MATCH_EVENTS_CLOSENESS_MS:
                    matched[cam2idx[cam]] = position[cam]
                    position[cam] += 1
            all_matched.append(matched)

        for cam in set(cameras) - {base_cam}:
            while position[cam] < n_events[cam]:
                single_event = [nan_val] * len(cameras)
                single_event[cam2idx[cam]] = position[cam]
                all_matched.append(single_event)
                position[cam] += 1

        return np.array(all_matched)

    def synchronize_with_parameters(self, cam1, cam2, offsets, parameters):
        # offsets e.g. {1: 82302.0, 3: 103713.0}
        # parameters e.g. {1: {'sensor_rows': 2625, 'mode_duration': 40}, ... }
        events_ij = self.get_matched_events([cam1, cam2], offsets, base_cam=cam1)
        # drift * t_f + shift + r / (R * fps)
        assert cam1 in parameters and cam2 in parameters
        p = parameters
        X = np.vstack((
            events_ij[cam2]['time'],
        )).T
        y = events_ij[cam1]['time'] + \
            events_ij[cam1]['start'] / p[cam1]['sensor_rows'] * p[cam1]['mode_duration_ms'] \
            - events_ij[cam2]['start'] / p[cam2]['sensor_rows'] * p[cam2]['mode_duration_ms']

        self.model[cam1] = {'time_per_row': float(p[cam1]['mode_duration_ms']) / p[cam1]['sensor_rows']}

        # model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        # model_ransac.fit(X, y)
        # shift = model_ransac.estimator_.intercept_[0]
        # coef = model_ransac.estimator_.coef_[0]
        model = linear_model.LinearRegression()
        model.fit(X, y)
        shift = model.intercept_
        coef = model.coef_
        self.model[cam2] = {'drift': coef[0],
                            'shift': shift,
                            'time_per_row': float(p[cam2]['mode_duration_ms']) / p[cam2]['sensor_rows']}

    def synchronize(self, cameras, offsets, base_cam=None):
        # offsets e.g. {1: 82302.0, 3: 103713.0}
        # parameters e.g. {1: {'sensor_rows': 2625, 'mode_duration': 40}, ... }
        if base_cam is None:
            base_cam = min(cameras)
        n = len(cameras) - 1
        y = []
        X = []
        for i, cam in enumerate(set(cameras) - {base_cam}):
            e = self.get_matched_events([base_cam, cam], offsets, base_cam)
            y_ = e[base_cam]['time'].reshape(-1, 1)
            X_ = np.zeros((len(e[cam]), 3 * n + 1))
            X_[:, 0] = - e[base_cam]['start']
            X_[:, 1 + i * 3 + 0] = e[cam]['time']
            X_[:, 1 + i * 3 + 1] = np.ones(len(e[cam]))
            X_[:, 1 + i * 3 + 2] = e[cam]['start']
            y.append(y_)
            X.append(X_)
        y = np.vstack(y)
        X = np.vstack(X)
        # model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(fit_intercept=False))
        # model_ransac.fit(X, y)
        # c = model_ransac.estimator_.coef_[0]
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(X, y)
        c = model.coef_[0]

        self.model = {}
        self.model[base_cam] = {'time_per_row': c[0]}
        for i, cam in enumerate(set(cameras) - {base_cam}):
            self.model[cam] = {'drift':     c[1 + i * 3 + 0],
                               'shift':        c[1 + i * 3 + 1],
                               'time_per_row': c[1 + i * 3 + 2],
                               }

    def get_time(self, cam, frame_time, row=None):
        assert cam in self.model
        if row is None:
            row = np.zeros_like(frame_time)
        drift = self.model[cam].get('drift', 1)
        shift = self.model[cam].get('shift', 0)
        time_per_row = self.model[cam]['time_per_row']

        return frame_time * drift + shift + row * time_per_row

    def __get_synchronized_frames_single_cam__(self, timestamps, master_timestamps, max_sync_error=None):
        if not max_sync_error:
            max_sync_error = scipy.stats.mode(np.diff(master_timestamps))[0] / 2  # half of the standard frame duration

        synchronized_timestamps = []
        synchronized_idx = []
        indices_after = np.searchsorted(timestamps, master_timestamps)
        try:
            for idx_after, t_master in zip(indices_after, master_timestamps):
                t_slave_before = timestamps[idx_after - 1]
                t_slave_after = timestamps[idx_after]
                if min(abs(t_slave_before - t_master), abs(t_slave_after - t_master)) > max_sync_error:
                    synchronized_timestamps.append(-1)
                    synchronized_idx.append(-1)
                elif abs(t_slave_before - t_master) < abs(t_slave_after - t_master):
                    synchronized_timestamps.append(t_slave_before)
                    synchronized_idx.append(idx_after - 1)
                else:
                    synchronized_timestamps.append(t_slave_after)
                    synchronized_idx.append(idx_after)
        except IndexError:
            pass

        return np.array(synchronized_timestamps), np.array(synchronized_idx)

    def get_synchronized_frames(self, timestamps, master=None, perfect_master=True, dropped=True, max_sync_error=None):
        assert self.model
        assert isinstance(timestamps, dict)

        if not master:
            master = min(timestamps.keys())
        cameras = timestamps.keys()
        master_timing = self.get_time(master, timestamps[master])
        if perfect_master:
            # perfect master timing (no dropped frames)
            cam_mode_ms, _ = scipy.stats.mode(np.diff(master_timing))
            ref_timing = np.arange(master_timing[0], master_timing[-1], cam_mode_ms)
        else:
            ref_timing = master_timing

        sync_timing = {}
        sync_frames = {}
        for cam in cameras:
            times, frames = self.__get_synchronized_frames_single_cam__(self.get_time(cam, timestamps[cam]),
                                                                        ref_timing, max_sync_error)
            sync_timing[cam] = times
            sync_frames[cam] = frames

        min_length = min([len(x) for x in sync_timing.values()])
        ref_timing = ref_timing[:min_length]
        for cam in cameras:
            sync_timing[cam] = sync_timing[cam][:min_length]
            sync_frames[cam] = sync_frames[cam][:min_length]

        sync_timing_array = np.vstack(sync_timing.values()).T
        sync_frames_array = np.vstack(sync_frames.values()).T.astype(int)

        if not dropped:
            no_dropped = np.all(sync_timing_array != -1, axis=1)
            sync_timing_array = sync_timing_array[no_dropped]
            sync_frames_array = sync_frames_array[no_dropped]
            ref_timing = ref_timing[no_dropped]

        return sync_timing_array, sync_frames_array, ref_timing

    def get_synchronized_image_sources(self, sources, master=None, perfect_master=True, dropped=True, max_sync_error=None):
        assert self.model
        cameras = sources.keys()
        timestamps = {cam: sources[cam].timestamps_ms for cam in cameras}
        timing, frames, ref_timing = self.get_synchronized_frames(timestamps, master, perfect_master,
                                                                  dropped, max_sync_error)
        synchronized_sources = {
            cam: imagesource.SynchronizedSource(
                sources[cam],
                frames[:, cameras.index(cam)],
                timing[:, cameras.index(cam)] - ref_timing)
            for cam in cameras}

        return synchronized_sources


if __name__ == '__main__':
    # example 4 camera synchronization
    cameras = [1, 2, 3, 4]
    filenames = {cam: 'data/ice_hockey/%d.mp4' % cam for cam in cameras}

    # load video files and extract frame timestamps
    sources = {cam: imagesource.TimedVideoSource(filenames[cam])
               for cam in cameras}
    for source in sources.itervalues():
        source.extract_timestamps()

    # detect flash events
    sync = FlashVideoSynchronization()
    sync.detect_flash_events(filenames)

    # # save all detected events for analysis
    # features = {cam: extract_features(filenames[cam], compute_luminance_median, dtype=np.uint8) for cam in cameras}
    # sync.save_event_images(sources, features, 'out/events')

    # manually set rough offset by matching an event
    sync.show_events()
    matching_events = {1: 3, 3: 2, 2: 8, 4: 2}
    offsets = {cam: sync.events[cam][matching_events[cam]]['time'] for cam in cameras}
    sync.show_events(offsets)  # now the events should appear aligned

    # # optionally filter bad events and fix some wrongly detected
    # images_dimensions = {}
    # for cam in cameras:
    #     img = sources[cam].get_image(0)
    #     images_dimensions[cam] = img.shape
    # obsolete_regions = {cam: {'top': 28, 'bottom': images_dimensions[cam][0] - 1}
    #                     for cam in cameras}
    # override_good = {0: [], 1: [1917, 5983, 10718], 2: [3605, 5122, 14935], 3: [6415, ], 4: []}  # force to stay
    # override_start = {0: [], 1: [(10718, 983), ], 2: [], 3: [(6415, 633), ], 4: []}  # force to stay and fix start
    # override_bad = {0: [], 1: [], 2: [12992, ], 3: [], 4: []}  # force to filter out
    # sync.filter_events(obsolete_regions, override_good, override_bad, override_start)

    # synchronize cameras: find parameters transformations that map camera time to reference camera time
    sync.synchronize(cameras, offsets, base_cam=1)
    print sync.model

    print sync.get_time(1, 0, 1000)

    # get frame synchronized image sources
    sources_sync = sync.get_synchronized_image_sources(sources, master=1, dropped=False)  # , perfect_master=False)

    # use the synchronized video sources to show synchronized frames with time deviations
    frame = 0
    for i, (cam, source) in enumerate(sources_sync.iteritems()):
        plt.figure()
        plt.title('err: %02.f ms' % source.get_synchronization_error(frame))
        img = source.get_image(frame)
        if img is not None:
            plt.imshow(img)
        plt.grid(False)
        plt.axis('off')
    plt.show()

