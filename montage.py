import numpy as np
import cv2


class Montage(object):
    """
    example:

    plt.figure()
    montage = Montage((2500, 1600), (2, 3))
    plt.imshow(montage.montage(list_of_images))
    plt.grid(False)
    plt.tight_layout(0)
    plt.axis('off')
    """
    def __init__(self, montage_size, nm=None):
        """
        Initialize image montage.

        :param montage_size: montage size in pixels (w, h)
        :type montage_size: tuple
        :param nm: number of images horizontally and vertically (n, m)
        :type nm: tuple
        """
        self.montage_size = np.array(montage_size)
        if nm is None:
            self.nm = None
            self.cell_size = None
        else:
            self.nm = np.array(nm, dtype=int)
            self.cell_size = self.montage_size / self.nm
        self.shape = None
        self.sizes = None

    def __adjust_image_size__(self, img_size):
        """
        Compute a new size for an image to fit into the montage.
        """
        ratio = img_size[0] / float(img_size[1])
        sizes_ratio = self.cell_size.astype(float) / img_size
        if sizes_ratio[0] < sizes_ratio[1]:
            # horizontally tight
            dst_size = [self.cell_size[0], self.cell_size[0] / ratio]
        else:
            # vertically tight
            dst_size = [self.cell_size[1] * ratio, self.cell_size[1]]
        return np.round(dst_size).astype(int)

    def montage(self, images):
        """
        Make a montage out of images.

        :param images: list of images, max n*m
        :type images: list of array-like images
        :return: image montage
        :rtype: array-like
        """
        if not self.sizes:
            self.sizes = [self.__adjust_image_size__(img.shape[:2][::-1]) for img in images]
        if not self.shape:
            if images[0].ndim == 2:
                self.shape = tuple(self.montage_size[::-1])
            else:
                self.shape = tuple(self.montage_size[::-1]) + (3,)

        images_resized = [cv2.resize(f, (tuple(np.round(s).astype(int)))) for f, s in zip(images, self.sizes)]
        out = np.zeros(self.shape, dtype=np.uint8)
        for i in range(len(images)):
            x = int(i % self.nm[0]) * int(self.cell_size[0])
            y = int(i / self.nm[0]) * int(self.cell_size[1])
            imgw, imgh = self.sizes[i]
            out[y: y + imgh, x: x + imgw] = images_resized[i]
        return out
