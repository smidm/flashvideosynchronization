import os
import logging
import itertools
import math
import matplotlib.pylab as plt
import matplotlib
import numpy as np
from numpy.lib.recfunctions import rec_drop_fields
import scipy
import cv2
import joblib
import yaml
import json
from sklearn import linear_model
from functools import reduce

try:
    import seaborn as sns

    has_seaborn = True
except ImportError:
    has_seaborn = False
import imagesource


logging.basicConfig(level=logging.INFO)
memory = joblib.Memory(location=".", verbose=2)


@memory.cache
def extract_features(filename, frame_start=0, frame_end=-1):
    image_source = imagesource.VideoSource(filename)
    return extract_features_from_source(image_source, frame_start, frame_end)


def extract_features_from_source(source, frame_start=0, frame_end=-1):
    source.color_conversion_from_bgr = cv2.COLOR_BGR2Lab
    features = []
    if frame_end == -1:
        if not math.isinf(source.frame_count):
            frame_end = int(source.frame_count)
            frame_range = range(frame_start, frame_end)
        else:
            frame_end = frame_start  # for logging
            frame_range = itertools.count(start=frame_start)
    else:
        frame_range = range(frame_start, frame_end)
    source.seek(frame_start)
    for i, frame in enumerate(frame_range):
        try:
            img = source.get_next_image()
        except IOError:
            break
        features.append(np.median(img[:, :, 0], axis=1))
        if (i % 10) == 0:
            logging.info("%d / %d" % (i, frame_end - frame_start))
    return np.array(features, dtype=np.uint8).T


def ramp_detection(profile, ramp_detection_thresh=4):
    max_pos = np.argmax(profile)
    try:
        start = np.flatnonzero(profile[:max_pos] > ramp_detection_thresh)[0]
    except IndexError:
        start = np.nan
    try:
        end = max_pos + np.flatnonzero(profile[max_pos:] > ramp_detection_thresh)[-1]
    except IndexError:
        end = np.nan
    return start, end


def detect_events(
    features2d,
    timestamps,
    hidden_scanlines=0,
    diff_max_peak_thresh=20,
    ramp_detection_thresh=4,
):
    diff = np.diff(features2d.astype(float), axis=1)
    height_px = features2d.shape[0]
    diff_max = np.max(diff, axis=0)
    idx = np.nonzero(diff_max > diff_max_peak_thresh)[0] + 1
    events = []
    for frame in idx:
        profile = diff[:, frame - 1]
        start, end = ramp_detection(profile, ramp_detection_thresh)
        if not np.isnan(start):
            events.append((frame, start, timestamps[frame], True))
        else:
            logging.warning("unable to detect event start in frame %d" % frame)
        if not np.isnan(end):
            events.append((frame, end, timestamps[frame], False))
        else:
            logging.warning("unable to detect event end in frame %d" % frame)

    events = np.array(
        events,
        dtype=[
            ("frame", int),
            ("position_px", float),
            ("frame_time", float),
            ("positive", bool),
        ],
    )

    # merge events split between two frames
    idx_bottom = np.nonzero(
        (events["position_px"] == (height_px - 1)) & (events["positive"] is False)
    )[0]
    idx_top = np.nonzero((events["position_px"] == 0) & (events["positive"] is True))[0]
    to_delete = []
    for idx in idx_bottom:
        if (
            idx + 1 < len(events)
            and events[idx]["frame"] + 1 == events[idx + 1]["frame"]
            and idx + 1 in idx_top
        ):
            to_delete.extend([idx, idx + 1])
    return np.delete(events, to_delete)


@memory.cache
def detect_events_in_video(filename, config=None):
    if not config:
        config = {
            "hidden_scanlines": 0,
            "diff_max_peak_thresh": 20,
            "ramp_detection_thresh": 4,
        }
    features = extract_features(filename)
    source = imagesource.TimedVideoSource(filename)
    source.extract_timestamps()
    events = detect_events(
        features,
        source.timestamps_ms,
        config["hidden_scanlines"],
        config["diff_max_peak_thresh"],
        config["ramp_detection_thresh"],
    )

    return events


class FlashVideoSynchronization(object):
    def __init__(self):
        self.events = (
            {}
        )  # dict of record arrays, dtype=[('frame', int), ('position_px', float),
        #                               ('frame_time', float), ('positive', np.bool)]
        self.model = (
            {}
        )  # {cam: {'shift': x in ms, 'time_per_row':  y in ms, 'drift': z}, ...}
        self.base_cam = None
        self.DIFFMAX_PEAK_THRESH = 20
        self.RAMP_DETECTION_DIFF = 4
        self.MATCH_EVENTS_CLOSENESS_MS = 80  # 35

    def __str__(self):
        s = ""
        model_description = self.model_description()
        for cam in self.model:
            s += "camera {}: {}\n".format(cam, model_description[cam])
        return s

    def model_description(self):
        model_description = {}
        for cam, model_params in self.model.items():
            s = ""
            if "shift" in model_params:
                s += "time offset {:.2f} s, sensor clock drift {:.4f}, ".format(
                    model_params["shift"] / 1000, model_params["drift"]
                )
            s += "time per sensor row {time_per_row:.3f} ms".format(**model_params)
            model_description[cam] = s
        return model_description

    def detect_flash_events(self, filenames):
        """
        Detect flash events in input video files.

        Store results in self.events.

        :param filenames: {cam: filename, ...}
        """
        # cashed using joblib.Memory
        config = {
            "hidden_scanlines": 0,
            "diff_max_peak_thresh": self.DIFFMAX_PEAK_THRESH,
            "ramp_detection_thresh": self.RAMP_DETECTION_DIFF,
        }
        self.events = {
            cam: detect_events_in_video(filename, config)
            for cam, filename in filenames.items()
        }

    def show_events(
        self, offsets=None, positive=True, negative=False, xticks_seconds=True, title=None
    ):
        """
        TODO: plot estimated sub-frame time instead of frame time (use frame duration and image height)
        """
        assert len(self.events)
        cameras = list(self.events.keys())
        if offsets is None:
            offsets = {cam: 0 for cam in cameras}
        fig = plt.figure()
        if title is not None:
            plt.title(title)
        if has_seaborn:
            colors = sns.color_palette(n_colors=len(cameras))
        else:
            colors = None

        if positive:
            event_frame_times = {
                cam: [
                    e["frame_time"] - offsets[cam]
                    for e in self.events[cam]
                    if e["positive"]
                ]
                for cam in cameras
            }
            plt.eventplot(
                list(event_frame_times.values()), colors=colors, linelengths=0.95
            )
        if negative:
            event_frame_times = {
                cam: [
                    e["frame_time"] - offsets[cam]
                    for e in self.events[cam]
                    if not e["positive"]
                ]
                for cam in cameras
            }
            plt.eventplot(
                event_frame_times.values(),
                colors=colors,
                linelengths=0.95,
                linestyles="dotted",
            )
        plt.yticks(np.arange(len(cameras)), cameras)
        plt.ylabel("cameras")
        max_time = max([max(t) for t in event_frame_times.values()])
        if xticks_seconds:
            plt.xticks(
                np.arange(0, max_time, step=50000),
                (np.arange(0, max_time, step=50000) / 1000).astype(int),
            )
            plt.xlabel("time in seconds")
        else:
            plt.xlabel("time in milliseconds")
        plt.tight_layout(pad=0)
        if has_seaborn:
            sns.despine(fig)

    def filter_events(
        self,
        img_heights_px,
        drop_events_on_top=False,
        drop_events_on_bottom=False,
        drop_longer_and_shorter=False,
        drop_positive=False,
        drop_negative=False,
        force_keep={},
        force_drop={},
        force_position={},
        obsolete_regions={},
    ):
        """
        Filter out wrongly detected events:

        - events shorted than 0.9 or longer than 1.1 of the median event length
        - events starting on the image top (partial events)
        - events ending on the image bottom (partial events)
        - events in override_bad

        Fix events starts according to override_start.

        :param img_heights_px: image height for all cameras, {cam: height_px, ... }
        :param drop_events_on_top: drop events starting on the top (first row or first row after obsolete region)
        :param drop_events_on_bottom: drop events ending at the bottom (last row or last row before obsolete region)
        :param drop_longer_and_shorter: drop events of nonstandard length, apply only to events that are not split
        :param force_keep: force events to NOT BE filtered, specify events by a record array with frame, position
                        and positivity combination, e.g. {cam: [(frame, positive), (frame, positive)...], cam: ... }
        :param force_drop: force events to BE filtered, {cam: [(frame, position_px), (frame, position_px), ...], cam: ... }
        :param force_position: override event position, {cam: [(frame, horizontal position in px), ...], cam: ... }
        :param obsolete_regions: ignored stripes on the top and/or image bottom,
                                 {cam: {'top': top_px, 'bottom': bot_px}, ...}
        """

        # compute median event length in px for the cameras with the same img height
        heights_px = set(img_heights_px.values())
        median_event_length_px = {}
        mask_events_not_split = {}
        events = {}
        for cam in self.events.keys():
            cam_events = self.events[cam]
            override_bad_mask = self.__queries2mask__(
                cam_events, force_drop[cam] if cam in force_drop else None
            )
            override_good_mask = self.__queries2mask__(
                cam_events, force_keep[cam] if cam in force_keep else None
            )

            # filter out events
            mask_bad = np.zeros(len(cam_events), dtype=bool)
            if drop_events_on_top:
                mask_bad |= (
                    cam_events["position_px"] <= obsolete_regions[cam]["top"]
                ) & cam_events["positive"]
            if drop_events_on_bottom:
                mask_bad |= (
                    cam_events["position_px"] >= img_heights_px[cam] - 1
                ) & ~cam_events["positive"]
            if drop_positive:
                mask_bad |= cam_events["positive"]
            if drop_negative:
                mask_bad |= ~cam_events["positive"]

            # if drop_longer_and_shorter:
            #     # apply filter only to the events that are not split (naturally shortened)
            #     event_length_px = median_event_length_px[img_heights_px[cam]]
            #     mask_bad |= mask_events_not_split[cam] & \
            #                 (((cam_events['end'] - cam_events['position_px']) < event_length_px * 0.9) |
            #                 ((cam_events['end'] - cam_events['position_px']) > event_length_px * 1.1))

            mask_bad |= override_bad_mask

            # force events to stay
            events[cam] = cam_events[~mask_bad | override_good_mask]
            # override event position
            if force_position and cam in force_position:
                for row in force_position[cam]:
                    query = rec_drop_fields(force_position[cam], ["position_px"])
                    idxs = np.nonzero(self.__queries2mask__(events[cam], query))[0]
                    if len(idxs) == 0:
                        logging.warning(
                            "force_position can"
                            "t find a matching event: %s" % str(query)
                        )
                    elif len(idxs) > 1:
                        logging.warning(
                            "force_position ambiguous match for query: %s" % str(query)
                        )
                    else:
                        events[cam][idxs[0]]["position_px"] = row["position_px"]

        self.events = events

    def __queries2mask__(self, table, queries):
        """
        Convert queries to table mask.

        :param table: record array
        :param queries: record array or None
        :return: boolean mask for table
        """
        if queries is not None and len(queries) > 0:
            # masked items in the list
            masks = []
            for row in queries:
                masks.append(
                    np.all([table[col] == row[col] for col in row.dtype.names], axis=0)
                )
            return reduce(np.logical_or, masks)
        else:
            # when queries not present, return empty (false) mask
            return np.zeros(len(table), dtype=bool)

    def save_event_images(
        self, sources, features, output_dir, cameras=None, frame_range=None
    ):
        if cameras is None:
            cameras = sources.keys()
        if frame_range is None:
            frame_range = (0, np.inf)
        fig_width_in = 4
        fig_height_in = fig_width_in * 0.7
        params = {
            "figure.figsize": [fig_width_in, fig_height_in],
            "figure.dpi": 80,
            "savefig.dpi": 150,
            "font.size": 5,
        }
        plt.rcParams.update(params)
        for cam in cameras:
            for e in self.events[cam]:
                if not (frame_range[0] <= e["frame"] <= frame_range[1]):
                    continue
                fig = plt.figure()
                fig.set_size_inches(
                    fig_width_in * 2, fig_height_in * 2 * 0.6, forward=True
                )
                self.plot_frame_with_profile(
                    sources[cam].get_image(e["frame"]),
                    e["frame"],
                    features[cam],
                    e["position_px"],
                    e["positive"],
                )
                plt.savefig(
                    os.path.join(
                        output_dir,
                        "c%s_f%d_%dpx.jpg" % (str(cam), e["frame"], e["position_px"]),
                    )
                )
                plt.close(fig)

    def plot_frame_with_profile(
        self, img, frame_nr, features, position_px=None, positive=None
    ):
        def plot_position_line(position_px, positive):
            if position_px is not None:
                plt.hlines(
                    position_px,
                    plt.xlim()[0],
                    plt.xlim()[1],
                    "r",
                    linestyles="dotted" if positive else "dashed",
                )

        def set_axes_and_legend(ax):
            ax.spines["left"].set_position("zero")
            plt.locator_params(axis="x", nbins=4)
            ax.axes.yaxis.set_ticks([])
            if has_seaborn:
                sns.despine(ax=ax)
            plt.legend(loc="upper right", fontsize="x-small")

        plt.axis("tight")
        gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[6, 1, 1.5])

        ax1 = plt.subplot(gs[0])
        height = img.shape[0]
        ax1.imshow(img)
        plt.title("frame $I_n$")
        plt.grid(False)
        plt.axis("off")
        plot_position_line(position_px, positive)

        ax = plt.subplot(gs[1], sharey=ax1)
        plt.title("median line\nintensity")
        plt.plot(features[:, frame_nr], range(height), label=r"$\mathrm{I}_n$")
        plt.plot(features[:, frame_nr - 1], range(height), label=r"$\mathrm{I}_{n-1}$")
        plot_position_line(position_px, positive)
        set_axes_and_legend(ax)

        ax = plt.subplot(gs[2], sharey=ax1)
        plt.title("median line\nintensity\ndifference")
        plt.plot(
            features[:, frame_nr].astype(float)
            - features[:, frame_nr - 1].astype(float),
            range(height),
            label=r"$\mathrm{I}_n - \mathrm{I}_{n-1}$",
        )
        plt.plot(
            features[:, frame_nr - 1].astype(float)
            - features[:, frame_nr - 2].astype(float),
            range(height),
            label=r"$\mathrm{I}_{n-1} - \mathrm{I}_{n-2}$",
        )
        plot_position_line(position_px, positive)
        set_axes_and_legend(ax)

        # plt.autoscale(tight=True)
        plt.tight_layout()

    def get_matched_events(self, cameras, offsets, base_cam=None):
        """
        Find corresponding events to the reference camera events.

        Events are matched when closer than self.MATCH_EVENTS_CLOSENESS_MS.

        TODO: use estimated sub-frame time for matching

        :param cameras: list of all cameras to use
        :param offsets: rough offsets between the cameras (e.g. based on frame times)
        :param base_cam: reference camera
        :return: dictionary of matched events for all cameras, e.g.:
                {0: array([(1200,  632.,  719.,   5003.33),
                           (2442,  186.,  719.,  10181.67)],
                          dtype=[('frame', '<i8'), ('position_px', '<f8'), ('frame_time', '<f8'), ('positive', '?')]),
                1: array([( 639,  533.,  719.,   5325.83),
                          (1260,  667.,  719.,  10501.67)],
                          dtype=[('frame', '<i8'), ('position_px', '<f8'), ('frame_time', '<f8'), ('positive', '?')])}
        """
        assert self.events
        if base_cam is None:
            base_cam = min(cameras)
        matched = self.__match_events__(cameras, offsets, base_cam)
        mask_full_match = ~np.isnan(matched.sum(axis=1))
        matched_events = {}
        for idx, cam in enumerate(cameras):
            matched_events[cam] = self.events[cam][
                matched[mask_full_match].astype(int)[:, idx]
            ]
        return matched_events

    def __match_events__(self, cameras, offsets, base_cam):
        """
        Find corresponding event indices to the reference camera events.

        Events are matched when closer than self.MATCH_EVENTS_CLOSENESS_MS.

        TODO: instead self.events[cam]['frame_time'] use sub frame estimate, this also solves the restriction
              of one event per frame (see the assert)

        :param cameras: list of all cameras to use
        :param offsets: rough offsets between the cameras (e.g. based on frame times)
        :param base_cam: reference camera
        :return: list of matched event indices or nans, e.g. [[0, 0], [1, 1], [2, 2], [3, nan], [4, nan]]
        """
        assert self.events
        nan_val = np.nan

        time_rel = {
            cam: [e["frame_time"] - offsets[cam] for e in self.events[cam]]
            for cam in cameras
        }
        n_events = {cam: len(time_rel[cam]) for cam in cameras}
        cam2idx = {cam: i for i, cam in enumerate(cameras)}
        for cam in cameras:
            assert np.all(np.diff(time_rel[cam]) >= 0)
        position = {cam: 0 for cam in cameras}
        all_matched = []
        while position[base_cam] < n_events[base_cam]:
            matched = [nan_val] * len(cameras)
            matched[cam2idx[base_cam]] = position[base_cam]
            position[base_cam] += 1
            for cam in set(cameras) - {base_cam}:
                while position[cam] < n_events[cam] and (
                    time_rel[cam][position[cam]]
                    < time_rel[base_cam][matched[cam2idx[base_cam]]]
                    - self.MATCH_EVENTS_CLOSENESS_MS
                ):
                    single_event = [nan_val] * len(cameras)
                    single_event[cam2idx[cam]] = position[cam]
                    all_matched.append(single_event)
                    position[cam] += 1
                if position[cam] != n_events[cam] and (
                    time_rel[cam][position[cam]]
                    < time_rel[base_cam][matched[cam2idx[base_cam]]]
                    + self.MATCH_EVENTS_CLOSENESS_MS
                ):
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
        X = np.vstack((events_ij[cam2]["frame_time"],)).T
        y = (
            events_ij[cam1]["frame_time"]
            + events_ij[cam1]["position_px"]
            / p[cam1]["sensor_rows"]
            * p[cam1]["mode_duration_ms"]
            - events_ij[cam2]["position_px"]
            / p[cam2]["sensor_rows"]
            * p[cam2]["mode_duration_ms"]
        )

        self.model[cam1] = {
            "time_per_row": float(p[cam1]["mode_duration_ms"]) / p[cam1]["sensor_rows"]
        }

        # model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        # model_ransac.fit(X, y)
        # shift = model_ransac.estimator_.intercept_[0]
        # coef = model_ransac.estimator_.coef_[0]
        model = linear_model.LinearRegression()
        model.fit(X, y)
        shift = model.intercept_
        coef = model.coef_
        self.model[cam2] = {
            "drift": coef[0],
            "shift": shift,
            "time_per_row": float(p[cam2]["mode_duration_ms"]) / p[cam2]["sensor_rows"],
        }

    def synchronize(self, cameras, offsets, base_cam=None):
        """
        Fit synchronization model.

        :param cameras: set of cameras to synchronize
        :param offsets: e.g. {1: 82302.0, 3: 103713.0}
        :param base_cam: reference camera, if None then the camera with the smallest index is used
        """
        if base_cam is None:
            base_cam = min(cameras)
        n = len(cameras) - 1
        y = []
        X = []
        for i, cam in enumerate(set(cameras) - {base_cam}):
            e = self.get_matched_events([base_cam, cam], offsets, base_cam)
            n_matched_events = len(e[cam])
            y_ = e[base_cam]["frame_time"].reshape(-1, 1)
            X_ = np.zeros((n_matched_events, 3 * n + 1))
            X_[:, 0] = -e[base_cam]["position_px"]
            X_[:, 1 + i * 3 + 0] = e[cam]["frame_time"]
            X_[:, 1 + i * 3 + 1] = np.ones(len(e[cam]))
            X_[:, 1 + i * 3 + 2] = e[cam]["position_px"]
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

        self.base_cam = base_cam
        self.model = {base_cam: {"time_per_row": float(c[0])}}
        for i, cam in enumerate(set(cameras) - {base_cam}):
            self.model[cam] = {
                "drift": float(c[1 + i * 3 + 0]),
                "shift": float(c[1 + i * 3 + 1]),
                "time_per_row": float(c[1 + i * 3 + 2]),
            }
        return X, y

    def get_time(self, cam, frame_time, row=None):
        """
        Return global time for a frame time and row of a particular camera.

        :param cam:
        :param frame_time:
        :param row:
        :return:
        """
        assert cam in self.model
        if row is None:
            row = np.zeros_like(frame_time)
        drift = self.model[cam].get("drift", 1)
        shift = self.model[cam].get("shift", 0)
        time_per_row = self.model[cam]["time_per_row"]

        return frame_time * drift + shift + row * time_per_row

    def get_frame_position(self, cam, timestamps_ms, synchronized_time):
        """
        Return frame / row position for a synchronized time.

        :param cam: camera
        :param timestamps_ms: camera timestamps
        :param synchronized_time: queried synchronized time
        :return: frame, frame time [ms], row [px]; all in camera cam
        """
        assert cam in self.model

        drift = self.model[cam].get("drift", 1)
        shift = self.model[cam].get("shift", 0)
        time_per_row = self.model[cam]["time_per_row"]
        idx = np.searchsorted(timestamps_ms, (synchronized_time - shift) / drift)
        frame_time = timestamps_ms[idx - 1]
        row = ((synchronized_time - shift) / drift - frame_time) / (
            time_per_row / drift
        )
        return idx - 1, frame_time, row

    def __get_synchronized_frames_single_cam__(
        self, timestamps, master_timestamps, max_sync_error=None
    ):
        """
        Return synchronized timestamps and frames for a single camera.

        When the synchronization error is bigger than max_sync_error (dropped frame), there will be -1 in place of
        corresponding timestamp and frame index.

        :param timestamps:
        :param master_timestamps:
        :param max_sync_error: maximum synchronization error in ms
        :return: synchronized_timestamps, synchronized_idx
        """
        if not max_sync_error:
            max_sync_error = (
                scipy.stats.mode(np.diff(master_timestamps))[0] / 2
            )  # half of the standard frame duration

        synchronized_timestamps = []
        synchronized_idx = []
        indices_after = np.searchsorted(timestamps, master_timestamps)
        for idx_after, t_master in zip(indices_after, master_timestamps):
            t_slave_before = timestamps[idx_after - 1]
            t_slave_after = (
                timestamps[idx_after] if idx_after < len(timestamps) else np.inf
            )
            if (
                min(abs(t_slave_before - t_master), abs(t_slave_after - t_master))
                > max_sync_error
            ):
                synchronized_timestamps.append(-1)
                synchronized_idx.append(-1)
            elif abs(t_slave_before - t_master) < abs(t_slave_after - t_master):
                synchronized_timestamps.append(t_slave_before)
                synchronized_idx.append(idx_after - 1)
            else:
                synchronized_timestamps.append(t_slave_after)
                synchronized_idx.append(idx_after)

        return np.array(synchronized_timestamps), np.array(synchronized_idx)

    def get_synchronized_frames(
        self,
        timestamps,
        master=None,
        perfect_master=True,
        dropped=True,
        max_sync_error=None,
    ):
        """
        Return table of synchronized frame timings and frame indices.

        :param timestamps: frame timestamps for all cameras, {cam1: np.array([t1, t2, t3, ...]), cam2: ..., ...}
        :param master: reference camera
        :param perfect_master: fix timing of the master camera to be perfectly stable, without dropped frames
        :param dropped: allowed dropped frames (-1 in output arrays)
        :param max_sync_error: maximum synchronization error between master camera timing and other camera timings,
                               by default half of the master camera frame duration
        :return: sync_timing_array - synchronized timing for all cameras,
                 sync_frames_array - synchronized frame indices for all cameras
                 ref_timing - reference timing of the master camera
        """
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
            times, frames = self.__get_synchronized_frames_single_cam__(
                self.get_time(cam, timestamps[cam]), ref_timing, max_sync_error
            )
            sync_timing[cam] = times
            sync_frames[cam] = frames

        sync_timing_array = np.vstack(tuple(sync_timing.values())).T
        sync_frames_array = np.vstack(tuple(sync_frames.values())).T.astype(int)

        if not dropped:
            no_dropped = np.all(sync_timing_array != -1, axis=1)
            sync_timing_array = sync_timing_array[no_dropped]
            sync_frames_array = sync_frames_array[no_dropped]
            ref_timing = ref_timing[no_dropped]

        return sync_timing_array, sync_frames_array, ref_timing

    def get_synchronized_image_sources(
        self,
        sources,
        master=None,
        perfect_master=False,
        dropped=True,
        max_sync_error=None,
    ):
        """
        Return synchronized image sources.

        :param sources: image source, see imagesource module
        :param master: reference camera
        :param perfect_master: fix timing of the master camera to be perfectly stable, without dropped frames
        :param dropped: allowed dropped frames (-1 in output arrays)
        :param max_sync_error: maximum synchronization error between master camera timing and other camera timings,
                               by default half of the master camera frame duration
        :return: dict of synchronzed image sources, {camera: imagesource.SynchronizedSource, ...}
        """
        assert self.model
        cameras = list(sources.keys())
        timestamps = {cam: sources[cam].timestamps_ms for cam in cameras}
        timing, frames, ref_timing = self.get_synchronized_frames(
            timestamps, master, perfect_master, dropped, max_sync_error
        )
        synchronized_sources = {
            cam: imagesource.SynchronizedSource(
                sources[cam],
                frames[:, cameras.index(cam)],
                timing[:, cameras.index(cam)] - ref_timing,
            )
            for cam in cameras
        }

        return synchronized_sources

    def to_json(self):
        return json.dumps({"model": self.model, "base_cam": self.base_cam})

    def to_yaml(self):
        return yaml.dump({"model": self.model, "base_cam": self.base_cam})

    def from_yaml(self, s):
        data = yaml.load(s)
        self.model = data["model"]
        self.base_cam = data["base_cam"]

    def from_json(self, s):
        data = json.loads(s)
        self.base_cam = data["base_cam"]
        self.model = {int(k): v for k, v in data["model"].items()}
