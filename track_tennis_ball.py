#!/usr/bin/env python3
"""Track a tennis ball in a fixed-camera video and draw a fading trail."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    radius: float
    score: float
    area: float


@dataclass
class TrackState:
    x: float
    y: float
    vx: float
    vy: float
    missed: int
    score: float
    hits: int
    path: list[tuple[float, float, float] | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input MP4 video")
    parser.add_argument("output", type=Path, help="Output MP4 with tracking overlay")
    parser.add_argument("--trail", type=int, default=24, help="Trail length in frames")
    parser.add_argument("--detector", choices=("tracknet", "classical"), default="tracknet",
                        help="Ball detector (default: tracknet)")
    parser.add_argument("--model", type=Path, default=Path("models/tracknet_weights.pth"),
                        help="TrackNet .pth weights")
    parser.add_argument("--compile-model", action="store_true",
                        help="Compile TrackNet for faster repeated CPU inference")
    parser.add_argument("--trajectory-in", type=Path,
                        help="Reuse a trajectory CSV and skip detection")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Create CSV/preview without rendering the video")
    parser.add_argument("--debug-csv", type=Path, help="Optional detected trajectory CSV")
    parser.add_argument("--preview", type=Path, help="Optional contact-sheet preview image")
    return parser.parse_args()


def video_metadata(video_path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    result = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return result


def tracknet_point(class_map: np.ndarray, scale_x: float, scale_y: float) -> tuple[float, float, float] | None:
    heatmap = class_map.reshape(360, 640).astype(np.uint8)
    binary = (heatmap > 127).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    components: list[tuple[float, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if 2 <= area <= 100:
            peak = float(heatmap[labels == index].max())
            components.append((peak + min(area, 20) * 1.5, index))
    if not components:
        return None
    _, best = max(components)
    x, y = centers[best]
    radius = max(4.0, 2.5 * math.sqrt(scale_x * scale_y))
    return float(x * scale_x), float(y * scale_y), radius


def tracknet_path(video_path: Path, model_path: Path, compile_model: bool = False) -> tuple[list[tuple[float, float, float] | None], dict[str, float]]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"TrackNet weights not found: {model_path}. Run `python download_model.py` first."
        )
    import torch
    from tracknet_model import BallTrackerNet

    torch.set_num_threads(min(8, max(1, os.cpu_count() or 4)))
    metadata = video_metadata(video_path)
    scale_x = float(metadata["width"]) / 640.0
    scale_y = float(metadata["height"]) / 360.0
    model = BallTrackerNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    for _ in range(2):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Video has fewer than three readable frames")
        frames.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))

    result: list[tuple[float, float, float] | None] = [None, None]
    frame_index = 2
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
            stacked = np.concatenate((frames[-1], frames[-2], frames[-3]), axis=2)
            tensor = torch.from_numpy(
                np.moveaxis(stacked.astype(np.float32) / 255.0, 2, 0)[None]
            )
            class_map = model(tensor).argmax(dim=1).cpu().numpy()[0]
            result.append(tracknet_point(class_map, scale_x, scale_y))
            frames.pop(0)
            frame_index += 1
            if frame_index % 30 == 0:
                print(f"TrackNet inference: {frame_index}/{int(metadata['frames'])}", flush=True)
    cap.release()
    return filter_tracknet_path(result, scale_x), metadata


def filter_tracknet_path(path: list[tuple[float, float, float] | None], scale: float) -> list[tuple[float, float, float] | None]:
    """Remove isolated network predictions while preserving fast ball motion."""
    result = list(path)
    max_step = 125.0 * scale
    for index in range(1, len(result) - 1):
        point = result[index]
        if point is None:
            continue
        prev_point = result[index - 1]
        next_point = result[index + 1]
        far_prev = prev_point is None or math.hypot(point[0] - prev_point[0], point[1] - prev_point[1]) > max_step
        far_next = next_point is None or math.hypot(point[0] - next_point[0], point[1] - next_point[1]) > max_step
        if far_prev and far_next:
            result[index] = None
    return clean_path(result, max_gap=4)


def candidate_mask(prev: np.ndarray, frame: np.ndarray, nxt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a mask of small, moving yellow-green regions and the HSV frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Tennis-ball felt ranges from yellow to chartreuse under indoor lighting.
    colour = cv2.inRange(hsv, (18, 48, 95), (48, 255, 255))

    d_prev = cv2.absdiff(frame, prev).max(axis=2).astype(np.uint8)
    d_next = cv2.absdiff(frame, nxt).max(axis=2).astype(np.uint8)
    moving = ((d_prev > 14) & (d_next > 14)).astype(np.uint8) * 255
    moving = cv2.dilate(moving, np.ones((3, 3), np.uint8), iterations=1)

    mask = cv2.bitwise_and(colour, moving)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask, hsv


def extract_candidates(prev: np.ndarray, frame: np.ndarray, nxt: np.ndarray) -> list[Candidate]:
    mask, hsv = candidate_mask(prev, frame, nxt)
    height, width = frame.shape[:2]
    scale = width / 3840.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Candidate] = []

    min_area = max(2.0, 3.0 * scale * scale)
    max_area = 1200.0 * scale * scale
    max_side = 90.0 * scale

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > max_side or h > max_side or w < 2 or h < 2:
            continue
        # Ignore ceiling and the near-camera floor, where the ball never appears.
        cy_box = y + h / 2
        if cy_box < height * 0.13 or cy_box > height * 0.84:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = x + w / 2, y + h / 2

        region_mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([[[x, y]]])
        cv2.drawContours(region_mask, [shifted], -1, 255, -1)
        pixels = hsv[y : y + h, x : x + w][region_mask > 0]
        if pixels.size == 0:
            continue
        mean_h, mean_s, mean_v = pixels.mean(axis=0)

        perimeter = max(cv2.arcLength(contour, True), 1.0)
        circularity = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
        aspect = min(w, h) / max(w, h)
        fill = min(1.0, area / max(float(w * h), 1.0))
        hue_score = max(0.0, 1.0 - abs(float(mean_h) - 31.0) / 18.0)
        saturation_score = min(1.0, float(mean_s) / 150.0)
        value_score = min(1.0, float(mean_v) / 190.0)
        compact_score = 0.45 * circularity + 0.30 * aspect + 0.25 * fill
        size_score = math.exp(-abs(math.log(max(area, 1.0) / (45.0 * scale * scale + 1e-6))) * 0.35)
        score = 1.4 * hue_score + saturation_score + value_score + compact_score + 0.7 * size_score
        radius = max(4.0 * scale, 0.5 * math.hypot(w, h))
        candidates.append(Candidate(cx, cy, radius, score, area))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:80]


def read_candidates(video_path: Path) -> tuple[list[list[Candidate]], dict[str, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, prev = cap.read()
    ok2, current = cap.read()
    if not ok or not ok2:
        raise RuntimeError("Video has fewer than two readable frames")

    all_candidates: list[list[Candidate]] = [[]]
    frame_index = 1
    while True:
        ok, nxt = cap.read()
        if not ok:
            break
        all_candidates.append(extract_candidates(prev, current, nxt))
        prev, current = current, nxt
        frame_index += 1
        if frame_index % 60 == 0:
            print(f"Detecting candidates: {frame_index}/{total}", flush=True)
    all_candidates.append([])
    cap.release()
    return all_candidates, {"fps": fps, "width": width, "height": height, "frames": len(all_candidates)}


def track_candidates(all_candidates: list[list[Candidate]], width: int) -> list[tuple[float, float, float] | None]:
    """Use a beam search with velocity prediction to choose one coherent ball path."""
    beam: list[TrackState] = []
    gate = width * 0.075
    max_missed = 10

    for frame_index, candidates in enumerate(all_candidates):
        next_states: list[TrackState] = []

        # New hypotheses let the tracker recover at the start of each rally.
        for cand in candidates[:18]:
            start_penalty = 7.0 + frame_index * 0.002
            next_states.append(
                TrackState(cand.x, cand.y, 0.0, 0.0, 0, cand.score - start_penalty, 1,
                           [None] * frame_index + [(cand.x, cand.y, cand.radius)])
            )

        for state in beam:
            pred_x = state.x + state.vx
            pred_y = state.y + state.vy
            local_gate = gate * (1.0 + 0.45 * state.missed) + 0.7 * math.hypot(state.vx, state.vy)
            matches: list[tuple[float, Candidate]] = []
            for cand in candidates:
                distance = math.hypot(cand.x - pred_x, cand.y - pred_y)
                if distance <= local_gate:
                    matches.append((distance, cand))
            matches.sort(key=lambda item: item[0])

            for distance, cand in matches[:7]:
                dt = state.missed + 1
                measured_vx = (cand.x - state.x) / dt
                measured_vy = (cand.y - state.y) / dt
                alpha = 0.55 if state.hits > 2 else 0.8
                vx = alpha * measured_vx + (1.0 - alpha) * state.vx
                vy = alpha * measured_vy + (1.0 - alpha) * state.vy
                accel = math.hypot(vx - state.vx, vy - state.vy)
                transition = 0.75 * distance / max(local_gate, 1.0) + 0.25 * accel / max(gate, 1.0)
                hit_bonus = 0.35 + min(0.8, math.hypot(vx, vy) / max(gate, 1.0))
                path = state.path + [None] * state.missed + [(cand.x, cand.y, cand.radius)]
                # state.path already includes missed frames; trim to the current frame count.
                path = path[:frame_index] + [(cand.x, cand.y, cand.radius)]
                next_states.append(
                    TrackState(cand.x, cand.y, vx, vy, 0,
                               state.score + cand.score + hit_bonus - transition,
                               state.hits + 1, path)
                )

            if state.missed < max_missed:
                next_states.append(
                    TrackState(pred_x, pred_y, state.vx * 0.98, state.vy * 0.98,
                               state.missed + 1, state.score - 1.15, state.hits,
                               state.path + [None])
                )

        # Deduplicate nearby hypotheses and keep the best-scoring beam.
        next_states.sort(key=lambda s: s.score + 0.06 * s.hits, reverse=True)
        beam = []
        for state in next_states:
            if any(math.hypot(state.x - kept.x, state.y - kept.y) < width * 0.006 and
                   state.missed == kept.missed for kept in beam):
                continue
            beam.append(state)
            if len(beam) >= 90:
                break

    if not beam:
        return [None] * len(all_candidates)
    best = max(beam, key=lambda s: s.score + 0.12 * s.hits)
    path = best.path[: len(all_candidates)]
    if len(path) < len(all_candidates):
        path.extend([None] * (len(all_candidates) - len(path)))
    return path


def clean_path(path: list[tuple[float, float, float] | None], max_gap: int = 9) -> list[tuple[float, float, float] | None]:
    cleaned = list(path)
    valid = [i for i, point in enumerate(cleaned) if point is not None]
    for left, right in zip(valid, valid[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            p0, p1 = cleaned[left], cleaned[right]
            assert p0 is not None and p1 is not None
            for offset in range(1, gap + 1):
                t = offset / (gap + 1)
                cleaned[left + offset] = tuple((1 - t) * a + t * b for a, b in zip(p0, p1))
    return cleaned


def draw_overlay(frame: np.ndarray, path: list[tuple[float, float, float] | None], index: int, trail: int) -> np.ndarray:
    result = frame.copy()
    overlay = frame.copy()
    start = max(0, index - trail + 1)
    points = [(i, path[i]) for i in range(start, index + 1) if path[i] is not None]

    for pair_index in range(1, len(points)):
        prev_i, prev_p = points[pair_index - 1]
        curr_i, curr_p = points[pair_index]
        if prev_p is None or curr_p is None or curr_i - prev_i > 3:
            continue
        age = index - curr_i
        strength = max(0.08, 1.0 - age / max(trail, 1))
        colour = (0, int(170 + 85 * strength), 255)
        thickness = max(2, int(3 + 9 * strength))
        cv2.line(overlay, (round(prev_p[0]), round(prev_p[1])),
                 (round(curr_p[0]), round(curr_p[1])), colour, thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.78, result, 0.22, 0, result)
    point = path[index]
    if point is not None:
        x, y, radius = point
        center = (round(x), round(y))
        ring = max(14, round(radius * 1.8))
        cv2.circle(result, center, ring, (0, 215, 255), 4, cv2.LINE_AA)
        cv2.circle(result, center, 5, (255, 255, 255), -1, cv2.LINE_AA)
        label_origin = (center[0] + ring + 10, center[1] - ring - 4)
        cv2.putText(result, "TENNIS BALL", label_origin, cv2.FONT_HERSHEY_DUPLEX,
                    0.72, (20, 20, 20), 5, cv2.LINE_AA)
        cv2.putText(result, "TENNIS BALL", label_origin, cv2.FONT_HERSHEY_DUPLEX,
                    0.72, (0, 235, 255), 2, cv2.LINE_AA)
    return result


def render_video(input_path: Path, output_path: Path, path: list[tuple[float, float, float] | None],
                 fps: float, width: int, height: int, trail: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the output video")

    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}", "-i", "-", "-i", str(input_path),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    cap = cv2.VideoCapture(str(input_path))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        process.stdin.write(draw_overlay(frame, path, frame_index, trail).tobytes())
        frame_index += 1
        if frame_index % 60 == 0:
            print(f"Rendering: {frame_index}/{len(path)}", flush=True)
    cap.release()
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")


def save_debug_csv(csv_path: Path, path: list[tuple[float, float, float] | None], fps: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_seconds", "x", "y", "radius", "detected"])
        for index, point in enumerate(path):
            if point is None:
                writer.writerow([index, f"{index / fps:.4f}", "", "", "", 0])
            else:
                writer.writerow([index, f"{index / fps:.4f}", *(f"{v:.2f}" for v in point), 1])


def load_debug_csv(csv_path: Path) -> list[tuple[float, float, float] | None]:
    path: list[tuple[float, float, float] | None] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["detected"] == "1":
                path.append((float(row["x"]), float(row["y"]), float(row["radius"])))
            else:
                path.append(None)
    return path


def save_preview(video_path: Path, preview_path: Path, path: list[tuple[float, float, float] | None], trail: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, max(0, total - 1), 12, dtype=int)
    images: list[np.ndarray] = []
    for index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        shown = draw_overlay(frame, path, int(index), trail)
        shown = cv2.resize(shown, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(shown, f"{index / (cap.get(cv2.CAP_PROP_FPS) or 30):.1f}s", (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        images.append(shown)
    cap.release()
    if not images:
        return
    while len(images) < 12:
        images.append(np.zeros_like(images[0]))
    rows = [np.hstack(images[i : i + 4]) for i in range(0, 12, 4)]
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), np.vstack(rows))


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.trajectory_in:
        metadata = video_metadata(args.input)
        path = load_debug_csv(args.trajectory_in)
    elif args.detector == "tracknet":
        path, metadata = tracknet_path(args.input, args.model, args.compile_model)
    else:
        candidates, metadata = read_candidates(args.input)
        raw_path = track_candidates(candidates, int(metadata["width"]))
        path = clean_path(raw_path)
    hits = sum(point is not None for point in path)
    print(f"Tracked ball in {hits}/{len(path)} frames ({100 * hits / max(len(path), 1):.1f}%)")
    if args.debug_csv:
        save_debug_csv(args.debug_csv, path, float(metadata["fps"]))
    if args.preview:
        save_preview(args.input, args.preview, path, args.trail)
    if not args.analyze_only:
        render_video(args.input, args.output, path, float(metadata["fps"]),
                     int(metadata["width"]), int(metadata["height"]), args.trail)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
