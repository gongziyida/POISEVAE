import torch

from .CUBSingleDatasets import CUBSentences, CUBImageFt
    
class CUB(torch.utils.data.Dataset):
    def __init__(self, img_data_dir, txt_data_dir, split, device, transform=None, return_idx=False, **kwargs):
        """split: 'train' or 'test' """
        super().__init__()
        self.CUBtxt = CUBSentences(txt_data_dir, split=split, transform=transform, **kwargs)
        self.CUBimg = CUBImageFt(img_data_dir, split=split, device=device)
        self.return_idx = return_idx
        
    def __len__(self):
        return len(self.CUBtxt) - 1
    
    def __getitem__(self, idx):
        txt = self.CUBtxt.__getitem__(idx)[0]
        img = self.CUBimg.__getitem__(idx // 10)
        if self.return_idx:
            return img, txt, idx
        else:
            return img, txt
        
