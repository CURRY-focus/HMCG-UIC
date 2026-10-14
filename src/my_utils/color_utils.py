import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorLoss(nn.Module):
    def __init__(
            self,
            lambda_chroma=1.0,
            lambda_mean=0.2,
            lambda_vq=0.1,
    ):
        super().__init__()
        self.lambda_chroma = lambda_chroma
        self.lambda_mean = lambda_mean
        self.lambda_vq = lambda_vq

    def _to_rgb01(self, image):
        return ((image.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    def _to_yuv(self, image):
        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = b - y
        v = r - y
        return torch.cat([y, u, v], dim=1)

    def forward(self, rec, gt, color_info=None):
        rec_rgb = self._to_rgb01(rec)
        gt_rgb = self._to_rgb01(gt.detach())

        rec_yuv = self._to_yuv(rec_rgb)
        gt_yuv = self._to_yuv(gt_rgb)

        loss_chroma = F.l1_loss(rec_yuv[:, 1:], gt_yuv[:, 1:])
        loss_mean = F.l1_loss(
            rec_rgb.mean(dim=(2, 3)),
            gt_rgb.mean(dim=(2, 3)),
        )
        loss_vq = rec.new_tensor(0.0)
        color_bpp = rec.new_tensor(0.0)
        if color_info is not None:
            if color_info.get("vq_loss") is not None:
                loss_vq = color_info["vq_loss"].to(device=rec.device, dtype=rec.dtype)
            if color_info.get("bpp") is not None:
                color_bpp = color_info["bpp"].to(device=rec.device, dtype=rec.dtype)

        loss = (
            self.lambda_chroma * loss_chroma
            + self.lambda_mean * loss_mean
            + self.lambda_vq * loss_vq
        )
        logs = {
            "color/chroma": loss_chroma.detach(),
            "color/mean": loss_mean.detach(),
            "color/vq": loss_vq.detach(),
            "color/bpp": color_bpp.detach(),
        }
        return loss, logs
