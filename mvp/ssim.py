from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )