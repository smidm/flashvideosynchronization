import numpy as np
import os
import cv2
from synchronization import MultipleVideoSynchronization
import parameters
import logging
import pickle
logging.basicConfig(level=logging.INFO)


def compute_features(image_sequence, camera, frame_start, frame_end):
    features = np.zeros((image_sequence.get_image(0, camera).shape[0], frame_end - frame_start), dtype=np.float16)

    for i, frame in enumerate(xrange(frame_start, frame_end)):
        img = image_sequence.get_image(frame, camera)
        features[:, i] = img.sum(axis=1) / img.shape[1]
        # np.max(img.sum(axis=1) / (img.shape[1] * 255. / 100))
        if (i % 10) == 0:
            logging.info("cam %d: %d / %d" % (camera, i, frame_end - frame_start))
    return features

sequence_length_sec = 600
ocred_timings = 'video/usa_rus/frame_timings.pkl'
root = '../data/ihwc2015/'
    # '../data/ihwc2015/video/usa_rus/'

out_dir = 'out/'
features_file = os.path.join(out_dir, 'flashes2d.pkl')
out_feature_images = os.path.join(out_dir, '%s_%d.png')

# /home/matej/prace/sport_tracking/git/experiments/2016-08-22_subframe_synchronization
p = parameters.Parameters('parameters.yaml')
# p.c['data_root'] = root
bgs = p.get_foreground_sequence()
# images = p.get_image_sequence()
sync = MultipleVideoSynchronization()
sync.load(os.path.join(p.c['data_root'], ocred_timings))
match_start = np.datetime64('1900-01-01T' + p.c['match_start'])

if os.path.exists(features_file):
    with open(features_file, 'rb') as fr:
        features = pickle.load(fr)
        features_start = pickle.load(fr)
else:
    if not os.path.exists():
        os.mkdir(out_dir)
    features = {}
    features_start = {}

    for cam in p.c['cameras']:
        start = np.searchsorted(sync.get_timings()[cam], match_start)
        end = np.searchsorted(sync.get_timings()[cam], match_start + np.timedelta64(sequence_length_sec, 's'))
        features_start[cam] = start
        features[cam] = compute_features(bgs, cam, start, end)

        img = cv2.normalize(features[cam].astype(float),
                            np.zeros_like(features[cam], dtype=float),
                            0, 255, cv2.NORM_MINMAX, dtype=8)
        cv2.imwrite(out_feature_images % (os.path.splitext(features_file)[0], cam), img)

    with open(features_file, 'wb') as fw:
        pickle.dump(features, fw)
        pickle.dump(features_start, fw)
