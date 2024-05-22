import logging as log
import os
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import click
from utils import (
    FaceDetector,
    FaceForensics,
    apply_bboxes,
    save_results,
    setup_logging,
)

torch.backends.cudnn.benchmark = False


@click.group()
def cli():
    pass


@click.command()
@click.option("--root", "-r", type=str, required=True, help="Root directory")
@click.option(
    "--target_root", "-tr", type=str, required=True, help="Target root directory"
)
@click.option("--include_original", "-orig", is_flag=True, help="Include original")
@click.option("--save_bbs", "-sbb", is_flag=True, help="Save bounding boxes")
@click.option("--save_lmks", "-slmks", is_flag=True, help="Save landmarks")
@click.option(
    "--methods",
    "-m",
    type=str,
    multiple=True,
    default=["NeuralTextures", "Face2Face", "Deepfakes", "FaceSwap"],
    help="Methods",
)
@click.option("--compression", "-c", type=str, default="c23", help="Compression")
@click.option("--device", "-d", type=str, required=True, help="Device", default="cuda:0")
@click.option("--batch_size", "-bs", type=int, default=64, help="Batch size")
@click.option("--threshold", "-t", type=float, default=0.95, help="Threshold")
@click.option("--increment", "-inc", type=float, default=0.01, help="Increment")
@click.option("--num_workers", "-nw", type=int, default=4, help="Number of workers")
@click.option("--md_output", "-mdo", type=str, help="Metadata output")
@click.option(
    "--logdir",
    "-ld",
    type=click.Path(),
    default="./logdir/",
    help="Log directory",
    required=False,
)
@click.option("--num_videos", "-nv", type=int, default=int(1e9), help="Number of videos")
@click.option("--ext", "-e", type=str, default=".png", help="Extension")
@click.option("--quality", "-q", type=int, default=75, help="Quality")
@click.option(
    "--delete_orig", "-dlto", is_flag=True, help="Delete original", default=False
)
@click.option(
    "--save_as", "-sa", type=str, required=True, help="Save as", default="frames"
)
@click.option("--metadata_csv", "-mdcsv", type=str, help="Metadata CSV")
def prepro(
    root,
    target_root,
    include_original,
    save_bbs,
    save_lmks,
    methods,
    compression,
    device,
    batch_size,
    threshold,
    increment,
    num_workers,
    md_output,
    logdir: Path,
    num_videos,
    ext,
    quality,
    delete_orig,
    save_as,
    metadata_csv,
):
    args = locals()
    setup_logging(logdir, args)

    multiple_face_videos = {}

    dataset = FaceForensics(
        args["root"],
        args["target_root"],
        [*args["methods"]],
        args["include_original"],
        args["compression"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args["num_workers"],
        shuffle=False,
        collate_fn=lambda x: x[0],
    )
    detector = FaceDetector(
        args["device"],
        args["batch_size"],
        args["threshold"],
        args["increment"],
    )

    last_save_thread = None
    for i, (
        video_path,
        video,
        fps,
        target_video_path,
        bboxes_path,
        lmks_path,
    ) in enumerate(tqdm(loader, total=len(dataset))):
        if i == args["num_videos"]:
            break

        if video is None:
            log.error(f"Video {video_path} could not be loaded")
            continue

        if target_video_path.exists():
            log.log(log.INFO, f"Video {video_path} already processed")
            continue

        boxes, lmks, scores = detector(video)
        num_boxes = sum([len(frame_boxes) for frame_boxes in boxes])
        if num_boxes == 0:
            log.error(f"Video {video_path} has no faces")
            continue
        log.log(log.INFO, f"Detected {num_boxes} faces")

        faces = apply_bboxes(video, boxes, scale=1.3)

        # handle multiple faces videos
        for i, frame_faces in enumerate(faces):
            if len(frame_faces) > 1:
                if video_path not in multiple_face_videos:
                    multiple_face_videos[video_path] = []
                multiple_face_videos[video_path].append((i, len(frame_faces)))

        if video_path in multiple_face_videos:
            log.warning(f"Video {video_path} has multiple faces")
            log.info(f"Skipping video")
            continue

        if last_save_thread is not None:
            last_save_thread.join()

        last_save_thread = threading.Thread(
            target=save_results,
            args=(
                faces,
                boxes,
                lmks,
                video_path,
                target_video_path,
                bboxes_path,
                lmks_path,
                args["save_bbs"],
                args["save_lmks"],
                fps,
            ),
            kwargs={
                "ext": args["ext"],
                "quality": args["quality"],
                "delete_orig": args["delete_orig"],
                "save_as": args["save_as"],
            },
        )
        last_save_thread.start()

    if args["md_output"] and len(multiple_face_videos) > 0:
        df = pd.DataFrame.from_dict(multiple_face_videos, orient="index")
        df.to_csv(
            os.path.join(args["target_root"], args["md_output"]),
            index_label="video_path",
        )

    if metadata_csv:
        handle_metadata(metadata_csv, target_root)

def handle_metadata(metadata_csv, target_root):
    root_path = Path(target_root)
    metadata = pd.read_csv(metadata_csv)
    samples = set(
        [p.parent.relative_to(root_path) for p in root_path.rglob("*.png")]
    )  # get unique paths of videos, not frames
    labels = [0 if "original" in x.parts[0] else 1 for x in samples]
    samples = list(zip(samples, labels))
    samples = pd.DataFrame(samples, columns=["c23_path", "bin_label"])
    samples["c23_path"] = samples["c23_path"].apply(lambda x: x.as_posix())
    samples = samples.merge(metadata, how="left", on="c23_path")
    samples["c23_path"] = samples["c23_path"].apply(
        lambda x: list((root_path / x).glob("*.png"))
    )
    samples = samples.explode("c23_path", ignore_index=True)
    samples["c23_path"] = samples["c23_path"].apply(
        lambda x: Path(x).relative_to(root_path).as_posix()
    )

    # add multiclass label column
    manipulations = ["NeuralTextures", "Face2Face", "Deepfakes", "FaceSwap"]
    man_to_label = {manipulations[i]: i + 1 for i in range(len(manipulations))}
    samples["mc_label"] = samples["c23_path"].apply(
        lambda x: (
            man_to_label[x.split("/")[1]] if x.split("/")[1] in manipulations else 0
        )
    )

    # rename c23_path to relative_path
    samples.rename(columns={"c23_path": "relative_path"}, inplace=True)
    samples = (
        samples.sample(frac=1)
        .reset_index(drop=True)
        .loc[:, ["relative_path", "bin_label", "mc_label", "split"]]
    )
    samples.to_csv("faceforensics_frames.csv", index=False, header=True, sep=" ")


cli.add_command(prepro)

if __name__ == "__main__":
    cli()
