#!/usr/bin/env python3
"""
YouTube Video Source Utility
============================

Provides access to video content via YouTube URLs with timestamps and ROI
(region-of-interest) cropping. No video files are downloaded, stored, or
redistributed — the evaluation pipeline accesses content through YouTube's
platform just as a human viewer would.

Metadata JSON format
--------------------
{
  "POCUS_of_the_Abdominal_Aorta": {
    "youtube_url": "https://www.youtube.com/watch?v=XXXXX",
    "start_time": 18.21,
    "end_time": 131.61,
    "roi": { "x": 0, "y": 0, "w": 1280, "h": 720 }
  },
  ...
}

Fields
~~~~~~
- youtube_url : Full YouTube URL for the source video.
- start_time  : The YouTube timestamp (seconds) where the local video content
                begins.  QA timestamp ``t`` maps to YouTube time
                ``start_time + t``.
- end_time    : The YouTube timestamp (seconds) where the local video content
                ends.  Used for bounds-checking / duration queries.
- roi         : (optional) Region-of-interest crop.  Supports two formats:
                1. **Pixel** coordinates: keys ``x, y, w, h`` (integers).
                2. **Proportional** coordinates: keys ``x_prop, y_prop,
                   w_prop, h_prop`` (floats 0–1, fractions of the YouTube
                   frame).  When omitted the full frame is used.

Usage
-----
    from video_source import YouTubeVideoSource

    source = YouTubeVideoSource("metadata.json")

    # Get a temporary clip for a specific time window
    clip_path = source.get_clip("POCUS_of_the_Abdominal_Aorta",
                                 time_start=10.0, time_end=25.0)
    # ... use clip_path (e.g. send to VLM) ...
    os.remove(clip_path)  # caller is responsible for cleanup
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ROI:
    """Region of interest — supports both pixel and proportional coords.

    When *proportional* is True, x/y/w/h are fractions of the source frame
    (0.0–1.0).  The crop filter uses ffmpeg's ``iw``/``ih`` expressions so
    no separate dimension probe is needed.
    """
    x: float
    y: float
    w: float
    h: float
    proportional: bool = False

    def to_crop_filter(self) -> str:
        """Return an ffmpeg crop filter string."""
        if self.proportional:
            # Use ffmpeg input-dimension expressions
            return (
                f"crop=iw*{self.w:.6f}:ih*{self.h:.6f}"
                f":iw*{self.x:.6f}:ih*{self.y:.6f}"
            )
        return f"crop={int(self.w)}:{int(self.h)}:{int(self.x)}:{int(self.y)}"


@dataclass
class VideoMetadataEntry:
    """Metadata for a single video mapping."""
    youtube_url: str
    start_time: float = 0.0
    end_time: float = 0.0
    roi: Optional[ROI] = None

    @property
    def time_offset(self) -> float:
        """Alias: QA timestamps are relative to start_time."""
        return self.start_time

    @property
    def duration(self) -> float:
        """Duration of the local video segment in the YouTube source."""
        return self.end_time - self.start_time

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoMetadataEntry":
        roi = None
        if "roi" in d and d["roi"]:
            r = d["roi"]
            if "x_prop" in r:
                # Proportional ROI (0–1 fractions of source frame)
                roi = ROI(
                    x=float(r["x_prop"]),
                    y=float(r["y_prop"]),
                    w=float(r["w_prop"]),
                    h=float(r["h_prop"]),
                    proportional=True,
                )
            else:
                # Absolute pixel coordinates
                roi = ROI(
                    x=float(r["x"]),
                    y=float(r["y"]),
                    w=float(r["w"]),
                    h=float(r["h"]),
                    proportional=False,
                )
        # Support both old (time_offset) and new (start_time/end_time) formats
        start = float(d.get("start_time", d.get("time_offset", 0.0)))
        end = float(d.get("end_time", 0.0))
        return cls(
            youtube_url=d["youtube_url"],
            start_time=start,
            end_time=end,
            roi=roi,
        )


# ============================================================================
# YOUTUBE VIDEO SOURCE
# ============================================================================

class YouTubeVideoSource:
    """Access YouTube video content on-the-fly for benchmark evaluation.

    * Resolves the direct streaming URL with ``yt-dlp`` (cached).
    * Uses ``ffmpeg`` to extract only the needed clip segment from the
      stream — no full video is ever downloaded or stored.
    * Optionally crops to a region of interest.
    """

    def __init__(self, metadata_path: str, cache_stream_urls: bool = True):
        """
        Parameters
        ----------
        metadata_path : str
            Path to the metadata JSON file.
        cache_stream_urls : bool
            Whether to cache resolved stream URLs (they expire after ~6 h on
            YouTube, but within a single evaluation run caching is safe).
        """
        self.metadata: Dict[str, VideoMetadataEntry] = {}
        self._stream_url_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._cache_enabled = cache_stream_urls

        self._load_metadata(metadata_path)

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_metadata(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = json.load(f)

        for name, entry in raw.items():
            self.metadata[name] = VideoMetadataEntry.from_dict(entry)

        print(f"📂 Loaded metadata for {len(self.metadata)} videos from {path}")

    def list_videos(self):
        """Return a list of video names available in the metadata."""
        return list(self.metadata.keys())

    def has_video(self, video_name: str) -> bool:
        return video_name in self.metadata

    # ------------------------------------------------------------------
    # Stream URL resolution
    # ------------------------------------------------------------------

    def _resolve_stream_url(self, youtube_url: str) -> str:
        """Use yt-dlp to obtain a direct stream URL for the best mp4 stream.

        yt-dlp is invoked with ``--get-url`` so that no file is written to
        disk.  We request the best combined mp4 format that includes video.
        """
        with self._cache_lock:
            if self._cache_enabled and youtube_url in self._stream_url_cache:
                return self._stream_url_cache[youtube_url]

        cmd = [
            "yt-dlp",
            "--get-url",
            "--remote-components", "ejs:github",
            # Prefer a single combined mp4 stream so ffmpeg can seek easily
            "-f", "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
            "--no-playlist",
            "--no-warnings",
            youtube_url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlp failed for {youtube_url}:\n{result.stderr.strip()}"
            )

        # yt-dlp may return multiple URLs (video + audio) separated by newlines
        # when format is video+audio merge.  We take the first (video) URL.
        stream_url = result.stdout.strip().splitlines()[0]

        with self._cache_lock:
            if self._cache_enabled:
                self._stream_url_cache[youtube_url] = stream_url

        return stream_url

    # ------------------------------------------------------------------
    # Video duration query
    # ------------------------------------------------------------------

    def get_video_duration(self, video_name: str) -> float:
        """Return the duration of the *local* video content in seconds.

        If ``end_time`` and ``start_time`` are set in the metadata the
        duration is computed directly (fast, no network call).  Otherwise
        falls back to querying the full YouTube video duration via yt-dlp.
        """
        entry = self.metadata.get(video_name)
        if entry is None:
            raise KeyError(f"No metadata entry for '{video_name}'")

        # Fast path: use metadata start/end
        if entry.end_time > entry.start_time:
            return entry.duration

        cmd = [
            "yt-dlp",
            "--print", "duration",
            "--remote-components", "ejs:github",
            "--no-playlist",
            "--no-warnings",
            entry.youtube_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            # Fallback: probe the stream URL directly
            try:
                stream_url = self._resolve_stream_url(entry.youtube_url)
                return self._probe_duration(stream_url)
            except Exception:
                return float("inf")

        try:
            return float(result.stdout.strip())
        except ValueError:
            return float("inf")

    @staticmethod
    def _probe_duration(stream_url: str) -> float:
        """Probe stream duration via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            stream_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
        return float("inf")

    # ------------------------------------------------------------------
    # Clip extraction (the core method)
    # ------------------------------------------------------------------

    def get_clip(
        self,
        video_name: str,
        time_start: float,
        time_end: float,
        output_path: Optional[str] = None,
        strip_audio: bool = True,
    ) -> str:
        """Extract a clip from a YouTube video without storing the full video.

        The clip is extracted by having ``ffmpeg`` read directly from the
        resolved YouTube stream URL.  Only the bytes corresponding to the
        requested time window are fetched from the network.

        Parameters
        ----------
        video_name : str
            Key into the metadata (e.g. ``"POCUS_of_the_Abdominal_Aorta"``).
        time_start, time_end : float
            Start / end in seconds (relative to the QA timestamps).
            ``time_offset`` from the metadata is added automatically.
        output_path : str | None
            Where to write the clip.  If *None*, a temporary file is created
            (the caller must delete it when done).
        strip_audio : bool
            Drop the audio track (default True, matching existing pipeline).

        Returns
        -------
        str
            Absolute path to the extracted clip file.
        """
        entry = self.metadata.get(video_name)
        if entry is None:
            raise KeyError(f"No metadata entry for '{video_name}'")

        # Map QA timestamps → absolute YouTube timestamps
        abs_start = max(0.0, time_start + entry.start_time)
        abs_end = time_end + entry.start_time

        # Clamp to the known end boundary if available
        if entry.end_time > 0:
            abs_end = min(abs_end, entry.end_time)

        duration = abs_end - abs_start
        if duration <= 0:
            raise ValueError(
                f"Invalid clip duration ({duration:.2f}s) for {video_name} "
                f"[{time_start}-{time_end}] + start_time {entry.start_time}"
            )

        # Resolve stream URL
        stream_url = self._resolve_stream_url(entry.youtube_url)

        # Prepare output path
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = tmp.name
            tmp.close()

        # Build ffmpeg command
        #   -ss before -i  → fast seek on the network stream
        #   -t duration     → read only what we need
        vf_filters = []
        if entry.roi:
            vf_filters.append(entry.roi.to_crop_filter())

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{abs_start:.3f}",
            "-i", stream_url,
            "-t", f"{duration:.3f}",
        ]

        if vf_filters:
            # When we apply video filters we must re-encode
            cmd += ["-vf", ",".join(vf_filters)]
            cmd += ["-c:v", "libx264", "-preset", "ultrafast"]
        else:
            # Try stream copy first (fastest, no re-encode)
            cmd += ["-c:v", "copy"]

        if strip_audio:
            cmd.append("-an")

        cmd += [
            "-avoid_negative_ts", "make_zero",
            "-loglevel", "error",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            # Fallback: force re-encoding (handles cases where stream copy
            # fails or the stream format doesn't support seeking well)
            print(f"    ⚠️  Fast extract failed, retrying with re-encode...",
                  flush=True)
            cmd_retry = [
                "ffmpeg", "-y",
                "-ss", f"{abs_start:.3f}",
                "-i", stream_url,
                "-t", f"{duration:.3f}",
            ]
            re_vf = list(vf_filters)  # keep ROI if any
            cmd_retry += ["-vf", ",".join(re_vf)] if re_vf else []
            cmd_retry += [
                "-c:v", "libx264", "-preset", "ultrafast",
            ]
            if strip_audio:
                cmd_retry.append("-an")
            cmd_retry += [
                "-avoid_negative_ts", "make_zero",
                "-loglevel", "error",
                output_path,
            ]
            result2 = subprocess.run(cmd_retry, capture_output=True, text=True,
                                     timeout=120)
            if result2.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg clip extraction failed for {video_name}:\n"
                    f"  First attempt: {result.stderr.strip()}\n"
                    f"  Retry:         {result2.stderr.strip()}"
                )

        return output_path

    # ------------------------------------------------------------------
    # Convenience: resolve video name from a question filename
    # ------------------------------------------------------------------

    @staticmethod
    def video_name_from_question_file(question_filename: str) -> str:
        """Derive the video name key from a question JSON filename.

        ``POCUS_of_the_Abdominal_Aorta.json`` → ``POCUS_of_the_Abdominal_Aorta``
        ``score_pred_POCUS_of_the_Abdominal_Aorta.json`` → ``POCUS_of_the_Abdominal_Aorta``
        """
        base = os.path.basename(question_filename)
        base = base.replace("score_pred_", "").replace(".json", "")
        return base


# ============================================================================
# CLI — quick test / validation
# ============================================================================

def main():
    """Quick smoke-test: resolve a stream URL and extract a short clip."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test YouTube video source utility"
    )
    parser.add_argument("metadata", help="Path to metadata JSON")
    parser.add_argument("--video", type=str, default=None,
                        help="Video name to test (default: first in metadata)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Clip start time (s)")
    parser.add_argument("--end", type=float, default=5.0,
                        help="Clip end time (s)")
    parser.add_argument("--output", type=str, default="test_clip.mp4",
                        help="Output file")
    args = parser.parse_args()

    source = YouTubeVideoSource(args.metadata)

    video_name = args.video or source.list_videos()[0]
    print(f"Testing with video: {video_name}")

    print(f"Querying video duration...")
    dur = source.get_video_duration(video_name)
    print(f"  Duration: {dur:.1f}s")

    print(f"Extracting clip [{args.start:.1f}s – {args.end:.1f}s] → {args.output}")
    clip = source.get_clip(video_name, args.start, args.end,
                           output_path=args.output)
    size_kb = os.path.getsize(clip) / 1024
    print(f"  ✅ Clip saved: {clip} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
