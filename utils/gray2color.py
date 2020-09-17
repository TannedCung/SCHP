import cv2
import numpy as np

class makeColor():
    def __init__(self, num_cls):
        """
        Put colors to a gray image by creating a pallete of colors
        """
        self.palette = self.get_palette(num_cls)

    def get_palette(self, num_cls):
        """ Returns the color map for visualizing the segmentation mask.
        Args:
            num_cls: Number of classes
        Returns:
            The color map
        """
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def G2C(self, old_img):
        h, w = old_img.shape 
        R = np.zeros((h,w), dtype=np.uint8)
        G = np.zeros((h,w), dtype=np.uint8)
        B = np.zeros((h,w), dtype=np.uint8)
        # for i in range(20):
        #     print("number: {} ({}, {}, {})".format(i, self.palette[3*i+0],self.palette[3*i+1],self.palette[3*i+2]))

        for c in range (int(len(self.palette)/3)):
            # print(int(len(self.palette)/3))
            r_chan = np.where(old_img==c, self.palette[3*c+0], 0)
            g_chan = np.where(old_img==c, self.palette[3*c+1], 0)
            b_chan = np.where(old_img==c, self.palette[3*c+2], 0)
            R = cv2.add(R, np.array(r_chan, dtype=np.uint8))
            G = cv2.add(G, np.array(g_chan, dtype=np.uint8))
            B = cv2.add(B, np.array(b_chan, dtype=np.uint8))
        new_img = cv2.merge((R,G,B))
        return np.array(new_img, dtype=np.uint8)