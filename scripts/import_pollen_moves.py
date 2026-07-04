"""Import Pollen Robotics' Reachy Mini move libraries as Sparky animations.

One-shot, network-required, run manually from the repo root:

    bot/.venv/bin/python scripts/import_pollen_moves.py [--only NAME]...
        [--limit N] [--verify] [--dry-run]

Sources (both Apache-2.0):
  - pollen-robotics/reachy-mini-emotions-library  (81 clips + ogg audio)
  - pollen-robotics/reachy-mini-dances-library    (19 clips, motion only)

Their format: {"description": str, "time": [s...@50Hz],
  "set_target_data": [{"head": 4x4 (translation in meters),
                       "antennas": [a, b] rad, "body_yaw": rad, ...}, ...]}

Sparky's format (bot/services/animation_player.py): frame_rate 24 +
per-channel keyframes, degrees, head positions in file-units (x10 = mm).
Conversion:
  head[:3,:3] -> R.as_euler("xyz", degrees=True)  (same axes string
      create_head_pose feeds scipy, so the round-trip is exact)
  head[:3,3] meters -> file-units x100 (player: file x10 = mm, then /1000)
  antennas[0] -> r_antenna_angle, antennas[1] -> l_antenna_angle (rad->deg;
      index-preserving through the same set_target path; matches nod's
      r=-20/l=+20 convention)
  body_yaw rad->deg -> body_angle
  resample 50Hz -> 24fps with np.interp after t -= t[0]

Audio (emotions only): soundfile decodes the ogg -> <name>_sfx.wav 24kHz
mono s16, which plays under the default ANIMATION_AUDIO=sfx mode.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent.parent
ANIM_DIR = REPO / "bot" / "animations"
DATASETS = [
    ("pollen-robotics/reachy-mini-emotions-library", True),   # with audio
    ("pollen-robotics/reachy-mini-dances-library", False),
]
FRAME_RATE = 24.0
M_TO_FILE_UNITS = 100.0     # meters -> file-units (file x10 = mm)
EULER_LIMIT_DEG = 60.0
POS_WARN_UNITS = 5.0
POS_REJECT_UNITS = 6.0
FRAME_DELTA_LIMIT_DEG = 30.0


def transcode_to_sfx_wav(audio_path: Path, out_path: Path) -> bool:
    """Ogg/Opus -> 24kHz mono s16 wav via soundfile (libsndfile) + numpy.
    (The system ffmpeg is an x86 binary that SIGKILLs on this machine.)"""
    import wave

    import soundfile as sf

    try:
        data, sr = sf.read(str(audio_path), dtype="float64", always_2d=True)
        mono = data.mean(axis=1)
        if sr != 24000:
            t_in = np.arange(len(mono)) / sr
            t_out = np.arange(int(len(mono) * 24000 / sr)) / 24000.0
            mono = np.interp(t_out, t_in, mono)
        pcm = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm.tobytes())
        return True
    except Exception as e:
        print(f"  audio transcode failed for {audio_path.name}: {e}")
        return False


def convert(source: dict, name: str) -> tuple[dict | None, list[str]]:
    """Pollen move dict -> Sparky clip dict (or None on hard reject)."""
    notes = []
    t = np.asarray(source["time"], dtype=np.float64)
    frames = source["set_target_data"]
    if len(t) != len(frames) or len(t) < 2:
        return None, [f"REJECT: malformed time/frames ({len(t)}/{len(frames)})"]
    t = t - t[0]

    heads = np.asarray([f["head"] for f in frames], dtype=np.float64)   # (N,4,4)
    eulers = R.from_matrix(heads[:, :3, :3]).as_euler("xyz", degrees=True)  # (N,3) r,p,y
    pos = heads[:, :3, 3] * M_TO_FILE_UNITS                              # (N,3)
    ants = np.asarray([f["antennas"] for f in frames], dtype=np.float64)
    ants_deg = np.rad2deg(ants)                                          # (N,2) [r,l]
    body = np.rad2deg(np.asarray([f.get("body_yaw", 0.0) for f in frames],
                                 dtype=np.float64))

    # sanity: euler magnitude and wrap glitches
    if np.abs(eulers).max() > EULER_LIMIT_DEG:
        return None, [f"REJECT: euler out of range (max {np.abs(eulers).max():.1f} deg)"]
    if np.abs(np.diff(eulers, axis=0)).max() > FRAME_DELTA_LIMIT_DEG:
        notes.append(f"WARN: euler frame delta up to "
                     f"{np.abs(np.diff(eulers, axis=0)).max():.1f} deg (wrap glitch?)")
    pmax = np.abs(pos).max()
    if pmax >= POS_REJECT_UNITS:
        return None, [f"REJECT: head position {pmax:.1f} file-units"]
    if pmax > POS_WARN_UNITS:
        notes.append(f"WARN: head position up to {pmax:.1f} file-units")

    duration = float(t[-1])
    if duration < 1.0:
        notes.append(f"WARN: short clip ({duration:.2f}s)")
    if duration > 30.0:
        return None, [f"REJECT: too long ({duration:.1f}s)"]

    t_out = np.arange(int(round(duration * FRAME_RATE)) + 1) / FRAME_RATE

    def resample(col: np.ndarray) -> np.ndarray:
        return np.interp(t_out, t, col)

    data = {
        "head_rotation": {
            "joints": ["neck_roll", "neck_pitch", "neck_yaw"],
            "frames": np.stack([resample(eulers[:, i]) for i in range(3)],
                               axis=1).round(6).tolist(),
        },
        "head_position": {
            "joints": ["head_x", "head_y", "head_z"],
            "frames": np.stack([resample(pos[:, i]) for i in range(3)],
                               axis=1).round(6).tolist(),
        },
        "r_antenna_angle": {"joints": "r_antenna_angle",
                            "frames": resample(ants_deg[:, 0]).round(6).tolist()},
        "l_antenna_angle": {"joints": "l_antenna_angle",
                            "frames": resample(ants_deg[:, 1]).round(6).tolist()},
        "body_angle": {"joints": "body_angle",
                       "frames": resample(body).round(6).tolist()},
    }
    clip = {
        "frame_rate": FRAME_RATE,
        "data": data,
        "description": source.get("description", ""),
        "source": "pollen-robotics (Apache-2.0), converted by scripts/import_pollen_moves.py",
    }
    notes.append(f"ok: {duration:.1f}s, {len(t_out)} frames, "
                 f"yaw±{np.abs(eulers[:,2]).max():.0f}° pos±{pmax:.1f}u")
    return clip, notes


def verify_roundtrip(source: dict, name: str) -> list[str]:
    """Converted clip sampled at its own grid must match source interpolation."""
    sys.path.insert(0, str(REPO / "bot"))
    from services.animation_player import AnimationClip

    clip_dict, _ = convert(source, name)
    if clip_dict is None:
        return [f"{name}: rejected, nothing to verify"]
    clip = AnimationClip(name, clip_dict["frame_rate"], clip_dict["data"])

    t = np.asarray(source["time"]) - source["time"][0]
    heads = np.asarray([f["head"] for f in source["set_target_data"]])
    ants = np.asarray([f["antennas"] for f in source["set_target_data"]])
    errs = []
    t_check = np.arange(int(round(t[-1] * FRAME_RATE)) + 1) / FRAME_RATE
    for tc in t_check[:: max(1, len(t_check) // 40)]:
        head_pose, antennas_rad, _ = clip.pose_at(float(tc))
        # source interpolated at tc
        src_pos = np.array([np.interp(tc, t, heads[:, i, 3]) for i in range(3)])
        src_eul = np.stack(
            [np.interp(tc, t, R.from_matrix(heads[:, :3, :3]).as_euler("xyz")[:, i])
             for i in range(3)], axis=0)
        src_ants = np.array([np.interp(tc, t, ants[:, i]) for i in range(2)])
        d_pos_mm = np.abs(head_pose[:3, 3] * 1000 - src_pos * 1000).max()
        clip_eul = R.from_matrix(head_pose[:3, :3]).as_euler("xyz")
        d_eul_deg = np.rad2deg(np.abs(clip_eul - src_eul)).max()
        d_ant = np.abs(antennas_rad - src_ants).max()
        if d_pos_mm > 1.0 or d_eul_deg > 0.5 or d_ant > 0.01:
            errs.append(f"{name} t={tc:.2f}: pos {d_pos_mm:.2f}mm "
                        f"eul {d_eul_deg:.3f}° ant {d_ant:.4f}rad")
    return errs or [f"{name}: round-trip OK"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download


    imported, skipped = [], []
    for repo_id, with_audio in DATASETS:
        root = Path(snapshot_download(repo_id, repo_type="dataset"))
        movs = sorted(p for p in root.rglob("*.json")
                      if p.name not in ("metadata.jsonl",)
                      and "time" in p.read_text()[:2000])
        print(f"\n== {repo_id}: {len(movs)} move files")
        for p in movs:
            name = p.stem
            if args.only and name not in args.only:
                continue
            if args.limit and len(imported) >= args.limit:
                break
            source = json.loads(p.read_text())
            if "set_target_data" not in source:
                skipped.append((name, "no set_target_data"))
                continue
            if (ANIM_DIR / name / f"{name}.json").exists():
                skipped.append((name, "already exists"))
                continue
            clip, notes = convert(source, name)
            for n in notes:
                print(f"  {name}: {n}")
            if clip is None:
                skipped.append((name, notes[0]))
                continue
            if args.verify:
                for line in verify_roundtrip(source, name):
                    print("   ", line)
            if args.dry_run:
                continue
            out = ANIM_DIR / name
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.json").write_text(json.dumps(clip))
            if with_audio:
                for ext in (".ogg", ".opus", ".wav", ".mp3"):
                    audio = p.with_suffix(ext)
                    if audio.exists():
                        transcode_to_sfx_wav(audio, out / f"{name}_sfx.wav")
                        break
            imported.append(name)

    if not args.dry_run and imported:
        (ANIM_DIR / "POLLEN_ATTRIBUTION.md").write_text(f"""# Pollen Robotics move libraries

The animation clips listed below were converted on {date.today()} from
Pollen Robotics' Reachy Mini move libraries, published under the
Apache License 2.0:

- https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library
- https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library

Conversion: 50Hz 4x4-pose recordings resampled to 24fps keyframe channels
by scripts/import_pollen_moves.py. Audio tracks transcoded from Ogg/Opus.

Clips: {", ".join(sorted(imported))}
""")

    print(f"\nImported {len(imported)} clips; skipped {len(skipped)}")
    for name, why in skipped:
        if "already exists" not in why:
            print(f"  skipped {name}: {why}")


if __name__ == "__main__":
    main()
