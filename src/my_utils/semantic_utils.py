import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLoss(nn.Module):
    def __init__(
            self,
            lambda_attention=1.0,
            lambda_boundary=0.2,
            lambda_vq=0.1,
            eps=1e-6,
    ):
        super().__init__()
        self.lambda_attention = lambda_attention
        self.lambda_boundary = lambda_boundary
        self.lambda_vq = lambda_vq
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _to_rgb01(self, image):
        return ((image.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    def _to_gray(self, image):
        image = self._to_rgb01(image)
        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _sobel(self, gray):
        sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def _normalize_per_image(self, value):
        flat = value.flatten(1)
        min_v = flat.amin(dim=1).view(-1, 1, 1, 1)
        max_v = flat.amax(dim=1).view(-1, 1, 1, 1)
        return (value - min_v) / (max_v - min_v).clamp_min(self.eps)

    def _semantic_importance_target(self, image):
        gray = self._to_gray(image)
        edge = self._normalize_per_image(self._sobel(gray))
        local_mean = F.avg_pool2d(gray, kernel_size=15, stride=1, padding=7)
        contrast = self._normalize_per_image((gray - local_mean).abs())
        target = self._normalize_per_image(0.6 * edge + 0.4 * contrast)
        return target.detach()

    def forward(self, rec, gt, semantic_info=None):
        loss = rec.new_tensor(0.0)
        loss_attention = rec.new_tensor(0.0)
        loss_boundary = rec.new_tensor(0.0)
        loss_vq = rec.new_tensor(0.0)
        semantic_bpp = rec.new_tensor(0.0)

        if semantic_info is not None:
            attention = semantic_info.get("attention")
            if attention is not None:
                target = self._semantic_importance_target(gt.detach())
                target = F.interpolate(target, size=attention.shape[-2:], mode="bilinear", align_corners=False)
                loss_attention = F.l1_loss(attention.float(), target.float())
                loss_boundary = F.l1_loss(
                    self._sobel(attention.float()),
                    self._sobel(target.float()),
                )

            if semantic_info.get("vq_loss") is not None:
                loss_vq = semantic_info["vq_loss"].to(device=rec.device, dtype=rec.dtype)
            if semantic_info.get("bpp") is not None:
                semantic_bpp = semantic_info["bpp"].to(device=rec.device, dtype=rec.dtype)

            loss = (
                loss
                + self.lambda_attention * loss_attention
                + self.lambda_boundary * loss_boundary
                + self.lambda_vq * loss_vq
            )

        logs = {
            "semantic/attention": loss_attention.detach(),
            "semantic/boundary": loss_boundary.detach(),
            "semantic/vq": loss_vq.detach(),
            "semantic/bpp": semantic_bpp.detach(),
        }
        return loss, logs
