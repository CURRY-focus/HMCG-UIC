import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    def __init__(self, lambda_sketch=1.0, lambda_edge=0.2, lambda_lap=0.1, eps=1e-6):
        super().__init__()
        self.lambda_sketch = lambda_sketch
        self.lambda_edge = lambda_edge
        self.lambda_lap = lambda_lap
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        lap = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("lap", lap)

    def _to_gray(self, image):
        image = ((image.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        if image.shape[1] == 1:
            return image

        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _sobel(self, gray):
        sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def _normalize_per_image(self, edge):
        denom = edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(self.eps)
        return edge / denom

    def _laplacian(self, gray):
        lap = self.lap.to(device=gray.device, dtype=gray.dtype)
        return F.conv2d(gray, lap, padding=1)

    def forward(self, rec, gt, structure_info=None):
        rec_gray = self._to_gray(rec)
        gt_gray = self._to_gray(gt.detach())

        rec_edge = self._normalize_per_image(self._sobel(rec_gray))
        gt_edge = self._normalize_per_image(self._sobel(gt_gray))
        loss_edge = F.l1_loss(rec_edge, gt_edge)

        loss_lap = F.l1_loss(self._laplacian(rec_gray), self._laplacian(gt_gray))

        loss_sketch = rec.new_tensor(0.0)
        if structure_info is not None and structure_info.get("sketch") is not None:
            sketch = structure_info["sketch"].float()
            sketch_target = F.interpolate(
                gt_edge.detach(),
                size=sketch.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss_sketch = F.l1_loss(sketch, sketch_target)

        loss = (
            self.lambda_sketch * loss_sketch
            + self.lambda_edge * loss_edge
            + self.lambda_lap * loss_lap
        )
        logs = {
            "structure/sketch": loss_sketch.detach(),
            "structure/edge": loss_edge.detach(),
            "structure/lap": loss_lap.detach(),
        }
        return loss, logs
