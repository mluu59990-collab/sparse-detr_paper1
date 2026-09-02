# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    elif image_set in ['val', 'test']:
            return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def _dedupe_paths(paths):
    seen = set()
    deduped = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _resolve_path(root, value):
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _split_aliases(image_set):
    aliases = {
        "train": ["train", "train2017"],
        "val": ["valid", "val", "val2017"],
        "test": ["test", "test2017"],
    }
    return aliases.get(image_set, [image_set])


def _image_folder_candidates(root, image_set):
    candidates = []
    for split in _split_aliases(image_set):
        candidates.extend([
            root / split / "images",
            root / split,
            root / "images" / split,
        ])

    split_root = root / image_set
    candidates.extend([
        split_root / "test" / "images",
        split_root / "valid" / "images",
        split_root / "val" / "images",
        split_root / "train" / "images",
        split_root / "JPEGImages",
        root / "images",
        root / "JPEGImages",
        root,
    ])
    return _dedupe_paths(candidates)


def _annotation_candidates(root, image_set):
    candidates = []
    names = [
        f"instances_{image_set}.json",
        f"{image_set}.json",
        "_annotations.coco.json",
        "annotations.json",
    ]

    for split in _split_aliases(image_set):
        split_names = [
            f"instances_{split}.json",
            f"instances_{image_set}.json",
            f"{split}.json",
            f"{image_set}.json",
            "_annotations.coco.json",
            "annotations.json",
        ]
        for name in split_names:
            candidates.extend([
                root / split / name,
                root / "annotations" / name,
            ])

    split_root = root / image_set
    for name in names:
        candidates.extend([
            split_root / name,
            root / "annotations" / name,
            root / name,
        ])

    for nested in ["test", "valid", "val", "train"]:
        for name in names:
            candidates.append(split_root / nested / name)

    return _dedupe_paths(candidates)


def _load_image_names(ann_file, limit=50):
    if not ann_file.exists():
        return []
    try:
        with ann_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    names = []
    for image in data.get("images", []):
        name = image.get("file_name")
        if name:
            names.append(Path(name))
        if len(names) >= limit:
            break
    return names


def _image_exists(img_folder, image_name):
    if image_name.is_absolute():
        return image_name.exists()
    return (img_folder / image_name).exists()


def _choose_annotation_file(root, image_set, explicit_ann_file):
    if explicit_ann_file is not None:
        return explicit_ann_file, [explicit_ann_file]

    candidates = _annotation_candidates(root, image_set)
    ann_file = next((p for p in candidates if p.exists()), candidates[0])
    return ann_file, candidates


def _choose_image_folder(root, image_set, ann_file, explicit_img_folder):
    if explicit_img_folder is not None:
        return explicit_img_folder, [explicit_img_folder], _load_image_names(ann_file)

    candidates = _image_folder_candidates(root, image_set)
    existing = [p for p in candidates if p.exists()]
    image_names = _load_image_names(ann_file)

    if existing and image_names:
        scored = [(sum(_image_exists(folder, name) for name in image_names), folder) for folder in existing]
        best_count, best_folder = max(scored, key=lambda item: item[0])
        if best_count > 0:
            return best_folder, candidates, image_names

    img_folder = existing[0] if existing else candidates[0]
    return img_folder, candidates, image_names


def _format_paths(paths):
    return ", ".join(str(path) for path in paths)


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    transform_set = 'train' if image_set == 'train' else 'val'

    explicit_img_folder = _resolve_path(root, getattr(args, "coco_img_folder", ""))
    explicit_ann_file = _resolve_path(root, getattr(args, "coco_ann_file", ""))

    ann_file, ann_candidates = _choose_annotation_file(root, image_set, explicit_ann_file)
    if not ann_file.exists():
        found_jsons = sorted(str(p) for p in (root / "annotations").glob("*.json")) if (root / "annotations").exists() else []
        found_jsons += sorted(str(p) for p in (root / image_set).rglob("*.json")) if (root / image_set).exists() else []
        found_jsons += sorted(str(p) for p in root.glob("*.json"))
        raise AssertionError(
            f'provided COCO annotation file {ann_file} does not exist. '
            f'Checked: {_format_paths(ann_candidates)}. '
            f'JSON files found: {found_jsons}'
        )

    img_folder, img_candidates, image_names = _choose_image_folder(root, image_set, ann_file, explicit_img_folder)
    assert img_folder.exists(), (
        f'provided COCO image folder {img_folder} does not exist. '
        f'Checked: {_format_paths(img_candidates)}'
    )

    if image_names and not any(_image_exists(img_folder, name) for name in image_names):
        examples = [str(name) for name in image_names[:5]]
        raise AssertionError(
            f'COCO image folder {img_folder} exists, but none of the sampled images from {ann_file} were found there. '
            f'Examples from annotation file: {examples}. '
            f'Pass --coco_img_folder with the directory that those file_name paths are relative to.'
        )

    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(transform_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
