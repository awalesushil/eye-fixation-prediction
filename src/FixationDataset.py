import os
import imageio

from torch.utils.data import Dataset

from utils import read_text_file


class FixationDataset(Dataset):
    def __init__(self, root_dir, image_file, fixation_file=None, transform=None, test=False):
        self.test = test
        self.root_dir = root_dir
        self.image_files = read_text_file(os.path.join(self.root_dir,image_file))[:100]
        self.image_transform = transform["image"]
        self.raw_transform = transform["raw_image"]
        
        if not self.test:
            self.fixation_files = read_text_file(os.path.join(self.root_dir,fixation_file))[:100]
            assert len(self.image_files) == len(self.fixation_files), "Lengths of image files and fixation files do not match!"
            self.fixation_transform = transform["fixation"]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)
        name = img_name.split("/")[-1]

        if not self.test:
            fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
            fix = imageio.imread(fix_name)
            sample = {"name":name, "image": image, "fixation": fix, "raw_image": image}
            if self.fixation_transform:
                sample["fixation"] = self.fixation_transform(sample["fixation"])
        else:
            sample = {"name":name, "image": image, "raw_image": image}

        if self.raw_transform:
            sample["raw_image"] = self.raw_transform(sample["raw_image"])

        if self.image_transform:
            sample["image"] = self.image_transform(sample["image"])
        
        return sample