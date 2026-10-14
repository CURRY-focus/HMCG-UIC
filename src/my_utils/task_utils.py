import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

def _split_plus(value, name):
    items = [item.strip() for item in value.split("+") if item.strip()]
    if len(items) == 0:
        raise ValueError(f"{name} must contain at least one item.")
    return items


def _split_optional_plus(value):
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split("+")]


def _broadcast_optional(items, count, name, default=""):
    if len(items) == 0:
        return [default for _ in range(count)]
    if len(items) == 1 and count > 1:
        return items * count
    if len(items) != count:
        raise ValueError(f"{name} must have one item or match machine_task_cv_type.")
    return items


def _parse_layer_weights(layer_weights):
    if layer_weights is None or layer_weights == "":
        return None
    weights = [float(item.strip()) for item in layer_weights.split(",") if item.strip()]
    if len(weights) == 0:
        raise ValueError("machine_task_layer_weights must contain at least one number.")
    return weights


def _clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = value
    return cleaned


class UnderwaterTaskAdapter(nn.Module):
    """Adapter for local underwater detection/saliency/segmentation models."""

    def __init__(
        self,
        task_type,
        model_class,
        model_path,
        output_type="feature",
        input_size=512,
        checkpoint_key="",
        model_kwargs="",
        normalize_type="",
        strict_load=False,
    ):
        super().__init__()
        if model_class == "":
            raise ValueError(f"{task_type} requires --machine_task_model_classes.")
        if model_path == "":
            raise ValueError(f"{task_type} requires --machine_task_model_paths.")

        self.task_type = task_type
        self.output_type = output_type
        self.input_size = int(input_size)
        self.model_class_lower = model_class.lower()
        self.model_options = json.loads(model_kwargs) if model_kwargs else {}
        self.normalize_type = normalize_type.lower()
        if self.normalize_type == "":
            self.normalize_type = "none" if model_class.lower() in {"ultralytics", "yolo"} else "imagenet"
        self.model = self._load_model(model_class, model_path, checkpoint_key, model_kwargs, strict_load)
        self.feature_names = list(getattr(self.model, "feature_names", []))
        self._feature_cache = {}
        self._feature_handles = []
        self._feature_layers = []
        self._feature_channels = []
        self._feature_strides = []
        if self.model_class_lower in {"ultralytics", "yolo"} and self.output_type == "feature":
            self._configure_yolo_feature_taps()
        self.model.eval()
        self.model.requires_grad_(False)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]), persistent=False)

    def forward(self, images, depth=None):
        images = F.interpolate(
            images * 0.5 + 0.5,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        if self.normalize_type == "imagenet":
            images = (images - self.image_mean[:, None, None]) / self.image_std[:, None, None]
        elif self.normalize_type not in {"none", "zero_one"}:
            raise ValueError("machine_task_normalize_types must use imagenet, none, or zero_one.")
        if depth is not None:
            depth = F.interpolate(
                depth * 0.5 + 0.5,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.normalize_type == "imagenet":
                depth = (depth - self.image_mean[:, None, None]) / self.image_std[:, None, None]
        if self.model_class_lower == "tc_usod":
            return self.model(images, depth)
        if self.model_class_lower in {"ultralytics", "yolo"} and self.output_type == "feature":
            self._feature_cache.clear()
            self.model(images)
            return self._collect_yolo_features()
        return self.model(images)

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def _load_model(self, model_class, model_path, checkpoint_key, model_kwargs, strict_load):
        model_class_lower = model_class.lower()
        if model_class_lower in {"ultralytics", "yolo"}:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError("Install ultralytics to use machine_task_model_classes='ultralytics'.") from exc
            return YOLO(model_path).model

        if model_class_lower == "tc_usod":
            return TCUSODFeatureModel(model_path, model_kwargs, output_type=self.output_type)

        raise ValueError(
            "The paper task supervisors require model classes 'ultralytics' and 'tc_usod'."
        )

    def _configure_yolo_feature_taps(self):
        self._feature_layers = self.model_options.get("feature_layers", [16, 19, 22])
        self._feature_channels = self.model_options.get("feature_channels", [])
        self._feature_strides = self.model_options.get("feature_strides", [8, 16, 32])
        if not all(isinstance(index, int) for index in self._feature_layers):
            raise ValueError("YOLO feature_layers must be a list of integer layer indices.")
        if len(self._feature_strides) != len(self._feature_layers):
            raise ValueError("YOLO feature_strides must match feature_layers.")
        if self._feature_channels and len(self._feature_channels) != len(self._feature_layers):
            raise ValueError("YOLO feature_channels must match feature_layers.")

        module_list = getattr(self.model, "model", None)
        if module_list is None or not hasattr(module_list, "__getitem__"):
            raise RuntimeError("The Ultralytics model does not expose its indexed module list.")
        if any(index < 0 or index >= len(module_list) for index in self._feature_layers):
            raise ValueError(
                f"YOLO feature layer indices {self._feature_layers} are invalid for a {len(module_list)}-layer model."
            )

        self.feature_names = [f"yolo_layer_{index}" for index in self._feature_layers]
        for index in self._feature_layers:
            handle = module_list[index].register_forward_hook(self._make_feature_hook(index))
            self._feature_handles.append(handle)

    def _make_feature_hook(self, index):
        def capture(_module, _inputs, output):
            self._feature_cache[index] = output

        return capture

    def _collect_yolo_features(self):
        features = []
        for position, (index, stride) in enumerate(zip(self._feature_layers, self._feature_strides)):
            if index not in self._feature_cache:
                raise RuntimeError(f"YOLO layer {index} did not produce a feature tensor.")
            feature = self._feature_cache[index]
            if not torch.is_tensor(feature) or feature.ndim != 4:
                raise RuntimeError(f"YOLO layer {index} must return a BCHW tensor.")
            expected_size = self.input_size // int(stride)
            if feature.shape[-2:] != (expected_size, expected_size):
                raise RuntimeError(
                    f"YOLO layer {index} returned spatial size {tuple(feature.shape[-2:])}; "
                    f"expected {(expected_size, expected_size)} for stride {stride}."
                )
            if self._feature_channels and feature.shape[1] != int(self._feature_channels[position]):
                raise RuntimeError(
                    f"YOLO layer {index} returned {feature.shape[1]} channels; "
                    f"expected {self._feature_channels[position]}."
                )
            features.append(feature)
        return tuple(features)


class TCUSODFeatureModel(nn.Module):
    def __init__(self, model_path, model_kwargs="", output_type="prediction"):
        super().__init__()
        kwargs = json.loads(model_kwargs) if model_kwargs else {}
        self.output_type = output_type
        repo_root = kwargs.get("repo_root")
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(model_path), "..", ".."))
        tc_usod_root = os.path.join(repo_root, "TC_USOD")
        pretrained_model = kwargs.get(
            "pretrained_model",
            os.path.join(repo_root, "pretrained_model", "80.7_T2T_ViT_t_14.pth.tar"),
        )

        if tc_usod_root not in sys.path:
            sys.path.insert(0, tc_usod_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from Models.USOD_Net import ImageDepthNet

        class Args:
            pass

        args = Args()
        args.pretrained_model = pretrained_model
        self.model = self._build_without_torchvision_download(ImageDepthNet, args)

        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(_clean_state_dict_keys(state_dict), strict=False)
        self.model.eval()
        self.model.requires_grad_(False)

        self.feature_layers = kwargs.get(
            "feature_layers",
            ["relu5d_2", "relu4d_2", "relu3d_2", "relu2d_2", "relu1d_m"],
        )
        self.feature_channels = kwargs.get("feature_channels", [128, 128, 128, 64, 16])
        self.feature_strides = kwargs.get("feature_strides", [16, 8, 4, 2, 1])
        self.feature_names = [f"tc_usod_decoder_{name}" for name in self.feature_layers]
        self._feature_cache = {}
        self._feature_handles = []
        if self.output_type == "feature":
            self._configure_feature_taps()

    def forward(self, images, depth=None):
        if depth is None:
            depth = self._pseudo_depth(images)
        if self.output_type == "feature":
            self._feature_cache.clear()
        outputs = self.model(images, depth)
        if self.output_type == "feature":
            return self._collect_features(images.shape[-2:])
        return outputs

    def _configure_feature_taps(self):
        if not (
            len(self.feature_layers) == len(self.feature_channels) == len(self.feature_strides)
        ):
            raise ValueError("TC-USOD feature_layers, feature_channels, and feature_strides must have equal lengths.")
        decoder = self.model.decoder
        for name in self.feature_layers:
            if not hasattr(decoder, name):
                raise ValueError(f"TC-USOD decoder has no feature layer named '{name}'.")
            handle = getattr(decoder, name).register_forward_hook(self._make_feature_hook(name))
            self._feature_handles.append(handle)

    def _make_feature_hook(self, name):
        def capture(_module, _inputs, output):
            self._feature_cache[name] = output

        return capture

    def _collect_features(self, input_size):
        if input_size[0] != input_size[1]:
            raise RuntimeError(f"TC-USOD expects square inputs, received {tuple(input_size)}.")
        features = []
        for name, channels, stride in zip(
            self.feature_layers,
            self.feature_channels,
            self.feature_strides,
        ):
            if name not in self._feature_cache:
                raise RuntimeError(f"TC-USOD decoder layer '{name}' did not produce a feature tensor.")
            feature = self._feature_cache[name]
            if not torch.is_tensor(feature) or feature.ndim != 4:
                raise RuntimeError(f"TC-USOD decoder layer '{name}' must return a BCHW tensor.")
            expected_size = input_size[0] // int(stride)
            expected_shape = (int(channels), expected_size, expected_size)
            if tuple(feature.shape[1:]) != expected_shape:
                raise RuntimeError(
                    f"TC-USOD decoder layer '{name}' returned {tuple(feature.shape[1:])}; "
                    f"expected {expected_shape}."
                )
            features.append(feature)
        return tuple(features)

    @staticmethod
    def _pseudo_depth(images):
        return (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        ).repeat(1, 3, 1, 1)

    @staticmethod
    def _build_without_torchvision_download(builder, args):
        try:
            import torchvision.models as tv_models
        except ImportError:
            return builder(args)

        original_resnet34 = tv_models.resnet34

        def resnet34_no_download(*model_args, **model_kwargs):
            model_kwargs.pop("pretrained", None)
            model_kwargs["weights"] = None
            return original_resnet34(*model_args, **model_kwargs)

        tv_models.resnet34 = resnet34_no_download
        try:
            return builder(args)
        finally:
            tv_models.resnet34 = original_resnet34


class MachineTaskLoss(nn.Module):
    """Frozen task-backbone feature loss for downstream-task-aware finetuning.

    The dataset used by this project only returns images, so this loss optimizes
    task feature consistency between the reconstruction and the original image
    without requiring detection boxes or segmentation masks.
    """

    def __init__(
        self,
        cv_type="det_coco",
        output_type="feature",
        loss_type="l1",
        normalize_features=True,
        layer_weights=None,
        model_paths=None,
        model_classes=None,
        input_sizes=None,
        checkpoint_keys=None,
        model_kwargs=None,
        normalize_types=None,
        strict_load=False,
    ):
        super().__init__()
        self.cv_types = _split_plus(cv_type, "machine_task_cv_type")
        output_types = _split_plus(output_type, "machine_task_output_type")
        if len(output_types) == 1 and len(self.cv_types) > 1:
            output_types = output_types * len(self.cv_types)
        if len(output_types) != len(self.cv_types):
            raise ValueError("machine_task_output_type must have one item or match machine_task_cv_type.")
        model_paths = _broadcast_optional(
            _split_optional_plus(model_paths),
            len(self.cv_types),
            "machine_task_model_paths",
        )
        model_classes = _broadcast_optional(
            _split_optional_plus(model_classes),
            len(self.cv_types),
            "machine_task_model_classes",
        )
        input_sizes = _broadcast_optional(
            _split_optional_plus(input_sizes),
            len(self.cv_types),
            "machine_task_input_sizes",
            default="512",
        )
        checkpoint_keys = _broadcast_optional(
            _split_optional_plus(checkpoint_keys),
            len(self.cv_types),
            "machine_task_checkpoint_keys",
        )
        model_kwargs = _broadcast_optional(
            _split_optional_plus(model_kwargs),
            len(self.cv_types),
            "machine_task_model_kwargs",
        )
        normalize_types = _broadcast_optional(
            _split_optional_plus(normalize_types),
            len(self.cv_types),
            "machine_task_normalize_types",
        )

        self.loss_type = loss_type
        self.normalize_features = normalize_features
        self.layer_weights = _parse_layer_weights(layer_weights)
        self.models = nn.ModuleList(
            [
                self._build_model(
                    cv_type_,
                    output_type_,
                    model_class_,
                    model_path_,
                    input_size_,
                    checkpoint_key_,
                    model_kwargs_,
                    normalize_type_,
                    strict_load,
                )
                for (
                    cv_type_,
                    output_type_,
                    model_class_,
                    model_path_,
                    input_size_,
                    checkpoint_key_,
                    model_kwargs_,
                    normalize_type_,
                ) in zip(
                    self.cv_types,
                    output_types,
                    model_classes,
                    model_paths,
                    input_sizes,
                    checkpoint_keys,
                    model_kwargs,
                    normalize_types,
                )
            ]
        )
        self.feature_names = []
        for model_index, model in enumerate(self.models):
            model_names = list(getattr(model, "feature_names", []))
            self.feature_names.extend(
                f"{self.cv_types[model_index]}/{name}" for name in model_names
            )
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _build_model(
        cv_type,
        output_type,
        model_class="",
        model_path="",
        input_size="512",
        checkpoint_key="",
        model_kwargs="",
        normalize_type="",
        strict_load=False,
    ):
        cv_type = cv_type.lower()
        output_type = output_type.lower()

        if cv_type in {"underwater_det", "underwater_saliency"}:
            return UnderwaterTaskAdapter(
                task_type=cv_type,
                model_class=model_class,
                model_path=model_path,
                output_type=output_type,
                input_size=input_size,
                checkpoint_key=checkpoint_key,
                model_kwargs=model_kwargs,
                normalize_type=normalize_type,
                strict_load=strict_load,
            )
        raise ValueError(
            f"Unsupported machine task cv_type '{cv_type}'. "
            "The paper configuration supports underwater_det and underwater_saliency."
        )

    def train(self, mode=True):
        super().train(False)
        for model in self.models:
            model.eval()
        return self

    def forward(self, rec, gt, depth=None):
        rec_features = self._flatten_features(self._extract(rec, depth))
        with torch.no_grad():
            depth_gt = depth.detach() if depth is not None else None
            gt_features = [feat.detach() for feat in self._flatten_features(self._extract(gt.detach(), depth_gt))]

        if len(rec_features) != len(gt_features):
            raise RuntimeError(
                f"Task feature count mismatch: rec={len(rec_features)}, gt={len(gt_features)}."
            )

        weights = self._get_weights(len(rec_features), rec.device, rec.dtype)
        loss = rec.new_tensor(0.0)
        use_feature_names = len(self.feature_names) == len(rec_features)
        details = {}
        for idx, (rec_feat, gt_feat, weight) in enumerate(zip(rec_features, gt_features, weights)):
            rec_feat, gt_feat = self._match_feature_shape(rec_feat, gt_feat)
            curr_loss = self._feature_distance(rec_feat, gt_feat)
            loss = loss + weight * curr_loss
            feature_name = self.feature_names[idx] if use_feature_names else f"layer_{idx}"
            details[f"machine_task/{feature_name}"] = curr_loss.detach()

        loss = loss / weights.sum().clamp_min(1e-6)
        details["machine_task/total"] = loss.detach()
        return loss, details

    def _extract(self, images, depth=None):
        features = []
        for model in self.models:
            if isinstance(model, UnderwaterTaskAdapter):
                features.append(model(images, depth))
            else:
                features.append(model(images))
        return features

    def _flatten_features(self, features):
        flattened = []
        for feat in features:
            if isinstance(feat, (list, tuple)):
                flattened.extend(self._flatten_features(feat))
            elif isinstance(feat, dict):
                flattened.extend(self._flatten_features([feat[key] for key in sorted(feat.keys())]))
            elif torch.is_tensor(feat):
                flattened.append(feat)
        if len(flattened) == 0:
            raise RuntimeError("The task model did not return any tensor features.")
        return flattened

    def _get_weights(self, num_features, device, dtype):
        if self.layer_weights is None:
            return torch.ones(num_features, device=device, dtype=dtype)
        if len(self.layer_weights) == 1:
            return torch.full((num_features,), self.layer_weights[0], device=device, dtype=dtype)
        if len(self.layer_weights) != num_features:
            raise ValueError(
                f"machine_task_layer_weights has {len(self.layer_weights)} values, "
                f"but the task backbone returned {num_features} feature tensors."
            )
        return torch.tensor(self.layer_weights, device=device, dtype=dtype)

    def _feature_distance(self, rec_feat, gt_feat):
        rec_feat = rec_feat.float()
        gt_feat = gt_feat.float()

        if self.normalize_features:
            norm_dim = self._norm_dim(rec_feat)
            rec_feat = F.normalize(rec_feat, p=2, dim=norm_dim, eps=1e-6)
            gt_feat = F.normalize(gt_feat, p=2, dim=norm_dim, eps=1e-6)

        if self.loss_type == "l1":
            return F.l1_loss(rec_feat, gt_feat)
        if self.loss_type == "mse":
            return F.mse_loss(rec_feat, gt_feat)
        if self.loss_type == "cosine":
            rec_flat = rec_feat.flatten(1)
            gt_flat = gt_feat.flatten(1)
            return 1.0 - F.cosine_similarity(rec_flat, gt_flat, dim=1).mean()

        raise ValueError("machine_task_loss_type must be one of: l1, mse, cosine.")

    @staticmethod
    def _norm_dim(feat):
        if feat.ndim == 3:
            return -1
        if feat.ndim >= 2:
            return 1
        return 0

    @staticmethod
    def _match_feature_shape(rec_feat, gt_feat):
        if rec_feat.shape == gt_feat.shape:
            return rec_feat, gt_feat
        if rec_feat.ndim == 4 and gt_feat.ndim == 4 and rec_feat.shape[:2] == gt_feat.shape[:2]:
            rec_feat = F.interpolate(rec_feat, size=gt_feat.shape[-2:], mode="bilinear", align_corners=False)
            return rec_feat, gt_feat
        raise RuntimeError(f"Task feature shape mismatch: rec={tuple(rec_feat.shape)}, gt={tuple(gt_feat.shape)}.")
