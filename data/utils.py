import json
import logging as log
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from batch_face import RetinaFace
from PIL import Image
from torch.utils.data import Dataset
from more_itertools import batched


class VideoDataset(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths

    def __getitem__(
        self, idx: int
    ) -> Tuple[Path, Union[torch.tensor, np.ndarray], Union[int, None]]:

        # use deepcopy to duplicate only one item on the list, not the entire list
        # see pytorch dataloader documentation and num_workers RAM consumption.
        video_path = deepcopy(self.video_paths[idx])
        log.log(log.INFO, f"Processing {video_path}")
        frames, fps = self.load_video_with_retries(video_path)
        return video_path, frames, fps

    def __len__(self) -> int:
        return len(self.video_paths)

    @classmethod
    def load_video_with_retries(
        cls, video_path: Path, max_attempts: int = 10
    ) -> Tuple[Union[np.ndarray, None], Union[int, None]]:
        try:
            for attempt in range(max_attempts):
                log.info(f"Loading video {video_path}, attempt {attempt + 1}")
                frames = []
                with open(video_path, "rb") as f:
                    vr = VideoReader(f, ctx=cpu(0))
                    fps = int(vr.get_avg_fps())
                    for i in range(len(vr)):
                        frame = vr[i]
                        frames.append(frame.asnumpy())
                log.log(log.INFO, f"Loaded {len(frames)} frames")
                if len(frames) == 0:
                    log.error(f"Video {video_path} has no frames")
                else:
                    break
            else:
                log.error(
                    f"Unable to load video {video_path} after {max_attempts} attempts"
                )
                return None, fps
                # frames = torch.stack([torch.from_numpy(frame) for frame in frames])
            return np.stack(frames), fps
        except Exception as e:
            log.error(f"Loading video {video_path} failed with exception: {str(e)}")
            return None, None

    @classmethod
    def video_iterator_with_retries(
        cls,
        video_path: Path,
        batch_size: int = 1000,
        max_attempts: int = 10,
    ) -> Union[np.ndarray, None]: # type: ignore
        
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)

            for batch_idx, frame_idxs in enumerate(
                batched(list(range(total_frames)), batch_size)
            ):
                for attempt in range(max_attempts):
                    log.info(
                        f"Loading video {video_path}, batch_idx {batch_idx}, attempt {attempt + 1}"
                    )
                    frames = vr.get_batch(frame_idxs)
                    frames = frames.asnumpy()

                    log.log(log.INFO, f"Loaded {len(frames)} frames")
                    if len(frames) == 0:
                        log.error(
                            f"Video {video_path}, batch_idx {batch_idx}, has no frames"
                        )
                    else:
                        break
                else:
                    log.eror(
                        f"Unable to load video {video_path}, batch_idx {batch_idx}, after {max_attempts} attempts"
                    )
                    yield None

                yield frames
        except Exception as e:
            log.error(
                f"Loading video {video_path}, failed with exception: {str(e)}"
            )
            yield None


class FaceForensics(VideoDataset):
    def __init__(
        self,
        root: str,
        target_root: str,
        methods: List[str],
        include_original: bool,
        compression: str,
    ):

        self.root = Path(root)
        self.target_root = Path(target_root)
        self.target_root.mkdir(exist_ok=True, parents=True)
        self.methods = methods
        self.include_original = include_original
        self.compression = compression

        if self.include_original:
            self.methods.append("original_sequences")

        self.video_paths = []
        for method in self.methods:
            if method == "original_sequences":
                method_root = self.root / "original_sequences"
            else:
                method_root = self.root / "manipulated_sequences" / method
            method_root = method_root / self.compression
            self.video_paths.extend(list(method_root.rglob("*.mp4")))

        # remove videos that have already been processed
        self.video_paths = [
            video_path
            for video_path in self.video_paths
            if not (
                Path(
                    str(video_path).replace(str(self.root), str(self.target_root))
                ).exists()
            )
        ]

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Path, Union[torch.tensor, np.ndarray, None], Union[int, None], Path, Path, Path]:

        # use deepcopy to duplicate only one item on the list, not the entire list
        # see pytorch dataloader documentation and num_workers RAM consumption.
        video_path = deepcopy(self.video_paths[idx])
        log.log(log.INFO, f"Processing {video_path}")
        target_video_path = Path(
            str(video_path).replace(str(self.root), str(self.target_root))
        )
        boxes_path = target_video_path.joinpath("bboxes.json")
        lmks_path = target_video_path.joinpath("landmarks.json")
        frames, fps = self.load_video_with_retries(video_path)

        return video_path, frames, fps, target_video_path, boxes_path, lmks_path


class FaceDetector:
    def __init__(self, device, batch_size, threshold=0.95, increment=0.1):
        self.device = device
        self.batch_size = batch_size
        self.threshold = threshold
        self.increment = increment
        self.detector = RetinaFace(-1 if device == "cpu" else int(device[-1]))

    @torch.no_grad()
    def _batch_detect(self, frames, threshold):
        boxes, landmarks, scores = [], [], []
        for idx in range(0, len(frames), self.batch_size):
            batch_frames = frames[idx : idx + self.batch_size]
            batch_results = self.detector(batch_frames)
            for frame_results in batch_results:
                valid_res = [res for res in frame_results if res[2] > threshold]
                frame_boxes, frame_landmarks, frame_scores = (
                    list(zip(*valid_res)) if len(valid_res) > 0 else ([], [], [])
                )
                boxes.append(np.vstack(frame_boxes) if len(frame_boxes) > 0 else [])
                landmarks.append(
                    np.vstack(frame_landmarks) if len(frame_landmarks) > 0 else []
                )
                scores.append(frame_scores)
        return boxes, landmarks, scores

    """
    checks if the face detections are continuous in between frames,
    e.g. 1: [face], 2: [], 3: [face]
    in that case we lower the threshold and try again
    """

    @staticmethod
    def _is_continuous(scores):
        for score in scores:
            if len(score) == 0:
                return False
        return True

    @torch.no_grad()
    def __call__(self, frames):
        if isinstance(self.threshold, float):
            return self._batch_detect(frames, self.threshold)
        elif isinstance(self.threshold, (list, tuple)):
            assert (
                self.threshold[0] > self.threshold[1] and len(self.threshold) == 2
            ), "invalid threshold value"
            # try multiple times to get continuous faces
            for threshold in np.arange(*self.threshold, step=-self.increment):
                boxes, lmks, scores = self._batch_detect(frames, threshold)
                torch.cuda.empty_cache()
                if self._is_continuous(scores):
                    break
        return boxes, lmks, scores


def save_results(
    faces: List[List[torch.tensor]],
    boxes: List[Union[List[np.array], None]],
    lmks: List[Union[List[np.array], None]],
    video_path: Path,
    target_video_path: Path,
    boxes_path: Path,
    lmks_path: Path,
    save_bbs: bool,
    save_lmks: bool,
    fps: int,
    ext: str = ".png",
    quality: int = 75,
    delete_orig: bool = False,
    save_as: str = "frames",
):
    target_video_path.mkdir(exist_ok=True, parents=True)

    if save_as == 'frames':
        params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        for i, frame_faces in enumerate(faces):
            for j, face in enumerate(frame_faces):
                # pad name with 5 zeros
                face_name = str(i).zfill(5) + f"_{j}" + ext
                cv2.imwrite(
                    str(target_video_path / face_name),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                    params=params if ext == ".webp" else None,
                )
    else:
        raise ValueError(f"Invalid save_as value {save_as}")
    del faces

    # save boxes as json
    if save_bbs:
        save_bbs_to_file(filename=boxes_path, boxes=boxes)
    del boxes

    # save boxes as json
    if save_lmks:
        save_lmks_to_file(filename=lmks_path, lmks=lmks)
    del lmks

    # delete video path to save space
    if delete_orig:
        video_path.unlink()


def save_lmks_to_file(filename, lmks):
    lmks = {
        i: frame_landmarks.tolist()
        if not isinstance(frame_landmarks, list)
        else frame_landmarks
        for i, frame_landmarks in enumerate(lmks)
    }
    with open(filename, "w") as f:
        json.dump(lmks, f)


def save_bbs_to_file(filename, boxes):
    boxes = {
        i: frame_boxes.tolist() if not isinstance(frame_boxes, list) else frame_boxes
        for i, frame_boxes in enumerate(boxes)
    }
    with open(filename, "w") as f:
        json.dump(boxes, f, indent=4)


def scale_bbox(bbox, height, width, scale_factor):
    left, top, right, bottom = bbox
    size_bb = int(max(right - left, bottom - top) * scale_factor)
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    # Check for out of bounds, x-y top left corner
    left = max(int(center_x - size_bb // 2), 0)
    top = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - left, size_bb)
    size_bb = min(height - top, size_bb)
    return left, top, left + size_bb, top + size_bb


def apply_bbox(image, bbox, scale_factor=None):
    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)
    bbox[bbox < 0] = 0
    bbox = np.around(bbox).astype(int)
    if scale_factor:
        bbox = scale_bbox(bbox, image.shape[0], image.shape[1], scale_factor)
    left, top, right, bottom = bbox
    face = image[top:bottom, left:right, :]
    return face


def apply_bboxes(frames, bboxes, scale=None) -> List[np.ndarray]:
    per_image_faces = []
    for i, bboxes in enumerate(bboxes):
        faces = []
        if bboxes is not None:
            for bbox in bboxes:
                face = apply_bbox(frames[i], bbox, scale_factor=scale)
                faces.append(face)
        per_image_faces.append(faces)
    return per_image_faces


def save_faces(
    faces: List[List[torch.tensor]],
    bboxes: List[Union[List[np.array], None]],
    video_path: Path,
    target_video_path: Path,
    bboxes_path: Path,
    save_bbs: bool,
    ext: str = ".png",
    **kwargs,
):
    target_video_path.mkdir(exist_ok=True, parents=True)
    for i, frame_faces in enumerate(faces):
        # if len(frame_faces) > 1:
        #     if str(video_path) in multiple_face_videos:
        #         multiple_face_videos[str(video_path)].append((i, len(frame_faces)))
        #     else:
        #         multiple_face_videos[str(video_path)] = [(i, len(frame_faces))]
        for j, face in enumerate(frame_faces):
            # pad name with 5 zeros
            face_name = str(i).zfill(5) + f"_{j}" + ext
            Image.fromarray(face).save(str(target_video_path / face_name), **kwargs)

    # save bboxes as json
    if save_bbs:
        save_bbs_to_file(filename=bboxes_path, bboxes=bboxes)


def setup_logging(logdir: Path, args):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(logdir, f"{current_datetime}.log")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log.basicConfig(
        filename=log_filename,
        level=log.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info(args)
