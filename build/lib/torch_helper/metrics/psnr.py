import numpy as np
from .measure import Metric
class PSNR(Metric):
    def __init__(self):
        super().__init__()

    def calculate_psnr(self, img1, img2):
        """Calculate PSNR (Peak Signal-to-Noise Ratio).
        Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        Args:
            img1 (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
        Returns:
            float: psnr result.
        """
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20. * np.log10(255. / np.sqrt(mse))

   
    def cal(self, inp, tar):
        if len(inp)==3:
            batch_size = 1
            inp, tar = [inp], [tar]
        else:
            batch_size = inp.shape[0]
        sum_v = 0
        for i in range(batch_size):
            sum_v += self.calculate_psnr(inp[i], tar[i])
        return sum_v * 1.0 / batch_size
    