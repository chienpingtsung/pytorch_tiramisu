import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CRACK500(Dataset):
    def __init__(self, path):
        super(CRACK500, self).__init__()

        self.image_path = os.path.join(path, 'train/')
        self.label_path = os.path.join(path, 'label/')

        self.names = [
            p[14:-4]
            for p in glob(os.path.join(self.image_path, '*/', '*.jpg'))
        ]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        image = Image.open("{}{}.jpg".format(self.image_path, self.names[item])).resize((800, 800))
        label = Image.open("{}{}.png".format(self.label_path, self.names[item])).resize((800, 800))

        sample = {
            'image': np.array(image).transpose((2, 0, 1)),
            'label': np.array(label)
        }

        return sample
