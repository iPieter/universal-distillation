from torch.utils.data import Dataset, DataLoader
import logging
import logging.config
import yaml
logger = logging.getLogger('dataloader')
logger.info("Logger is started")

class JITTokenizedDataset(Dataset):
    """
    Pytorch Dataset that tokenizes a textual dataset just in time (JIT). With HuggingFace's fast tokenizers,
    this should not be an issue on a reasonably fast CPU.

    For Universal Distillation, multiple tokenizations are required and the results are aligned.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        logger.debug("Loading data")
        print("test")

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype("float").reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample