# Sub-millisecond Accurate Multiple Video Synchronization Using Camera Flashes

A python module for multiple video synchronization using photographic camera flashes. The module synchronizes arbitrary number of video sequences when there are present abrupt lighting changes that affect majority of pixels (e.g. flashes). The video sequences do not need to be overlapping. 

Sequences acquired using rolling shutter image sensors (vast majority of CMOS image sensors) can be synchronized up to sub-millisecond accuracy.

Outputs:

- sub-frame synchronized time for a frame, row pair (for rolling shutter)
- synchronized video sequences (frames with minimal temporal distances from all input sequences)

For details, sample synchronized videos and published paper see [project page](http://cmp.felk.cvut.cz/~smidm/rolling-shutter-camera-synchronization-with-sub-millisecond-accuracy.html).

## Installation and Running

Using a system package manager install:

- numpy

Compile or install:

- OpenCV 3.x with ffmpeg

```
$ pip install git+https://github.com/smidm/flashvideosynchronization.git
```

## Quickstart

```python
import imagesource
import flashvideosynchronization

cameras = [1, 2, 3, 4]
filenames = {cam: 'data/%d.mp4' % cam for cam in cameras}

# load video files and extract frame timestamps
sources = {cam: imagesource.TimedVideoSource(filenames[cam])
           for cam in cameras}
for source in sources.itervalues():
    source.extract_timestamps()

sync = flashvideosynchronization.FlashVideoSynchronization()
sync.detect_flash_events(filenames)

# manually set rough offset by matching an event
sync.show_events()
matching_events = {1: 3, 3: 2, 2: 8, 4: 2}
offsets = {cam: sync.events[cam][matching_events[cam]]['time'] for cam in cameras}
sync.show_events(offsets)  # now the events should appear aligned

# synchronize cameras: find parameters transformations that map camera time to reference camera time
sync.synchronize(cameras, offsets, base_cam=1)

# get sub-frame sychronized time for camera 1, frame 10 and row 100
print sync.get_time(cam=1, frame_time=sources[1].timestamps_ms[10], row=100)

# get frame synchronized image sources
sources_sync = sync.get_synchronized_image_sources(sources, master=1, dropped=False)

# show synchronized frames
frame = 10
for cam in cameras:
    plt.figure()
    plt.imshow(sources_sync[cam].get_image(frame))
```

The computationally demanding functions `FlashVideoSynchronization.detect_flash_events()` and `extract_features()` are cached using [joblib](https://pythonhosted.org/joblib/). Until the inputs change, they are computed only once. The cache is stored in `./joblib` directory.

For more examples see: https://github.com/smidm/flashvideosynchronization-notebook.

## Your Video Sequences

I would like to evaluate the software on more multi-view video sequences with abrupt lighting changes. If you can provide your data, please contact me at http://cmp.felk.cvut.cz/~smidm/.

## Cite

M. Šmíd and J. Matas, “Rolling Shutter Camera Synchronization with Sub-millisecond Accuracy,” in VISAPP - 12th International Conference on Computer Vision Theory and Applications, 2017.

