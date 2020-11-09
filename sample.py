import imagesource
from flashvideosynchronization import FlashVideoSynchronization
import matplotlib.pylab as plt

# example 3 camera synchronization
cameras = [1, 2, 3]
filenames = {cam: 'sample_data/%d.mp4' % cam for cam in cameras}

# load video files and extract frame timestamps
sources = {cam: imagesource.TimedVideoSource(filenames[cam])
           for cam in cameras}
for source in sources.values():
    source.extract_timestamps()

# detect flash events
sync = FlashVideoSynchronization()
sync.detect_flash_events(filenames)

# # save all detected events for analysis
# features = {cam: extract_features(filenames[cam], compute_luminance_median, dtype=np.uint8) for cam in cameras}
# sync.save_event_images(sources, features, 'out/events')

# manually set rough offset by matching an event
sync.show_events()
matching_events = {1: 0, 2: 0, 3: 0}
offsets = {cam: sync.events[cam][matching_events[cam]]['frame_time'] for cam in cameras}
sync.show_events(offsets)  # now the events should appear aligned

# synchronize cameras: find parameters transformations that map camera time to reference camera time
sync.synchronize(cameras, offsets, base_cam=1)
print(sync.model)

print(sync.get_time(1, 0, 1000))

# get frame synchronized image sources
sources_sync = sync.get_synchronized_image_sources(sources, master=1, dropped=False)  # , perfect_master=False)

# use the synchronized video sources to show synchronized frames with time deviations
frame = 0
for i, (cam, source) in enumerate(sources_sync.items()):
    plt.figure()
    plt.title('err: %02.f ms' % source.get_synchronization_error(frame))
    img = source.get_image(frame)
    if img is not None:
        plt.imshow(img)
    plt.grid(False)
    plt.axis('off')
plt.show()
