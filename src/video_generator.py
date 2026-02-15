"""
Create videos for all songs in the Album folder.
Each video uses Background.png as the static image, the WAV as audio,
and generates timed SRT subtitles from the lyrics text files.
Gold karaoke-style word-by-word highlights via FancyText.
"""
import os
import re
import wave
import json
import struct
import tempfile
import textwrap
from pathlib import Path

from PIL import Image as PILImage
if not hasattr(PILImage, 'ANTIALIAS'):
    PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS

try:
    from moviepy.editor import (
        ImageClip, AudioFileClip, CompositeVideoClip, VideoClip
    )
except ModuleNotFoundError:
    from moviepy import (
        ImageClip, AudioFileClip, CompositeVideoClip, VideoClip
    )

# Whisper utilities for voice recognition
from whisper_utils import (
    transcribe_with_whisper,
    align_lyrics_to_whisper,
    has_whisper as _HAS_WHISPER_FN,
)
_HAS_WHISPER = _HAS_WHISPER_FN()

from fancy_text import (
    create_word_fancytext_adv,
    create_static_fancytext,
)
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

ALBUM_DIR = Path("input")
OUTPUT_DIR = Path("output")
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 24

# Subtitle styling
FONT_SIZE = 55
FONT_PATH = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts', 'Guttural.ttf')
SUBTITLE_Y_RATIO = 0.72

# Global subtitle offset (seconds).  Negative = subtitles appear EARLIER.
# No longer needed — timing is generated from audio analysis, not human-timed SRT.
SUBTITLE_OFFSET = 0.0

# Pre-roll: show subtitle this many seconds BEFORE the detected vocal onset.
# Keeps it tight so the first highlighted word appears close to when the
# singer actually starts.  0.20 s gives a tiny read-ahead without lag.
SUBTITLE_PRE_ROLL = 0.20


def get_wav_duration(wav_path: str) -> float:
    """Get duration of a WAV file in seconds."""
    with wave.open(wav_path, 'r') as w:
        return w.getnframes() / float(w.getframerate())


def parse_lyrics(txt_path: str) -> list[dict]:
    """
    Parse lyrics from a text file.
    Returns a list of dicts with 'text' and 'is_direction' keys.
    Stage directions like [Verse], [Chorus], [Guitar Solo] are marked.
    Empty lines are treated as breath/pause markers.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            lines.append({'text': '', 'is_direction': False, 'is_blank': True})
        elif re.match(r'^\[.*\]$', stripped):
            # Stage direction like [Verse], [Chorus], [Guitar Solo Jazzy Opeth]
            lines.append({'text': stripped, 'is_direction': True, 'is_blank': False})
        elif re.match(r'^(Verse\s*(VI|IV|V?I{0,3})|Final Verse|Chorus|Bridge|Outro|Intro)\s*(\(.*\))?\s*:?\s*$', stripped, re.IGNORECASE):
            # Unbracketed directions like "Verse II:" or "Final Verse (Clean, building):"
            lines.append({'text': stripped, 'is_direction': True, 'is_blank': False})
        else:
            # Strip any inline [bracketed] text from lyric lines
            cleaned = re.sub(r'\[.*?\]', '', stripped).strip()
            if not cleaned:
                # Line was entirely bracketed text — treat as blank
                lines.append({'text': '', 'is_direction': False, 'is_blank': True})
            else:
                lines.append({'text': cleaned, 'is_direction': False, 'is_blank': False})

    return lines


# ── Lyrics-driven timing generation ─────────────────────────────

_INSTRUMENTAL_KEYWORDS = frozenset([
    'solo', 'instrumental', 'interlude', 'break', 'riff', 'intro',
    'outro', 'breakdown',
])


def _is_instrumental_direction(text: str) -> bool:
    """Return True if a direction marker means 'no vocals for a while'."""
    lower = text.lower()
    return any(kw in lower for kw in _INSTRUMENTAL_KEYWORDS)


def _detect_audio_start(wav_path: str, threshold_ratio: float = 0.15) -> float:
    """
    Detect when the audio actually gets loud (instruments start playing).
    Returns time in seconds.
    """
    energy, win_sec = compute_energy_profile(wav_path, window_sec=0.05,
                                              smooth_sec=0.5)
    if not energy:
        return 0.0

    peak = max(energy)
    threshold = peak * threshold_ratio
    for i, e in enumerate(energy):
        if e > threshold:
            return max(0, i * win_sec)
    return 0.0


def generate_lyrics_timing(lyrics_lines: list[dict], wav_path: str,
                           total_duration: float,
                           pre_roll: float = 0.20,
                           min_onset_gap: float = 1.4,
                           min_duration: float = 1.4) -> list[dict]:
    """
    Generate subtitle timing from .txt lyrics matched to the audio.

    Strategy — spectral voice-activity detection (VAD)
    --------------------------------------------------
    Instead of relying on broadband energy (which cannot distinguish vocals
    from loud guitars), we perform spectral analysis to find time regions
    where **singing is actually present**:

      1.  Compute a per-frame vocal-activity score using three features:
          • vocal-band energy ratio (300–4 000 Hz vs full spectrum)
          • spectral flux in vocal range (syllable changes)
          • amplitude-modulation depth in vocal band (singing ≈ 3–7 Hz)
      2.  Threshold the score (Otsu) → list of vocal segments.
      3.  Parse lyrics into sections; separate vocal vs instrumental.
      4.  Map instrumental markers to longest non-vocal gaps.
      5.  Distribute vocal lyrics proportionally across detected vocal
          segments.  Within each segment, lines are spaced by word count.
    """
    console.print("    Detecting vocal segments via spectral analysis...")

    # ── 1. Split lyrics into sections ────────────────────────────
    sections: list[dict] = []
    current_section = None

    for line in lyrics_lines:
        if line.get('is_direction'):
            if current_section is not None:
                sections.append(current_section)
            current_section = {
                'marker': line['text'],
                'lines': [],
                'is_instrumental': _is_instrumental_direction(line['text']),
            }
        elif line.get('is_blank'):
            if current_section is not None:
                current_section['lines'].append(
                    {'text': '', 'type': 'pause', 'word_count': 0})
        else:
            if current_section is None:
                current_section = {
                    'marker': '',
                    'lines': [],
                    'is_instrumental': False,
                }
            wc = len(line['text'].split())
            current_section['lines'].append(
                {'text': line['text'], 'type': 'lyric', 'word_count': wc})

    if current_section is not None:
        sections.append(current_section)
    if not sections:
        return []

    for sec in sections:
        sec['singing_lines'] = [l for l in sec['lines'] if l['type'] == 'lyric']
        sec['total_words'] = sum(l['word_count'] for l in sec['singing_lines'])

    console.print(f"    {len(sections)} sections")
    for sec in sections:
        tag = '  [instrumental]' if sec['is_instrumental'] else ''
        console.print(f"      {sec['marker'] or '(intro)':25s} "
                      f"{len(sec['singing_lines']):2d} lines, "
                      f"{sec['total_words']:3d} words{tag}")

    # ── 2. Detect vocal segments from audio ──────────────────────
    vocal_segs, vocal_score, hop_sec = detect_vocal_segments(wav_path)

    if not vocal_segs:
        console.print("    [yellow]No vocal segments detected — falling back to proportional[/yellow]")
        return _proportional_fallback(sections, total_duration)

    total_vocal_time = sum(s['end'] - s['start'] for s in vocal_segs)
    console.print(f"    Detected {len(vocal_segs)} vocal segments "
                  f"({total_vocal_time:.1f}s vocal / {total_duration:.1f}s total):")
    for vs in vocal_segs:
        console.print(f"      {vs['start']:6.1f}s – {vs['end']:6.1f}s  "
                      f"({vs['end'] - vs['start']:.1f}s)")

    # ── 2a. Detect phrase onsets for onset-driven timing ─────────
    raw_onsets, raw_strengths = detect_phrase_onsets(wav_path)
    # Filter onsets to within vocal segments
    vocal_onsets: list[float] = []
    for t in raw_onsets:
        for seg in vocal_segs:
            if seg['start'] - 0.3 <= t <= seg['end'] + 0.3:
                vocal_onsets.append(t)
                break
    # Inject vocal-segment starts as potential onsets
    for seg in vocal_segs:
        t0 = seg['start'] + 0.1
        if all(abs(t0 - o) > 0.8 for o in vocal_onsets):
            vocal_onsets.append(t0)
    vocal_onsets.sort()
    console.print(f"    Vocal onsets: {len(raw_onsets)} raw -> {len(vocal_onsets)} in vocal segs")

    # ── 2b. Force-split vocal segments at energy dips for instrumentals ──
    # Spectral VAD can mistake guitar solos for vocals (similar spectra).
    # Use broadband energy dips to find structural breaks, then split
    # any vocal segment that spans such a dip.
    has_instrumentals = any(s['is_instrumental'] for s in sections)
    if has_instrumentals:
        energy_profile, win_sec = compute_energy_profile(wav_path)
        if energy_profile:
            energy_dips = detect_energy_dips(energy_profile, win_sec,
                                            total_duration, min_dip_sec=2.0)
            # Keep only substantial dips (depth * duration > 3)
            big_dips = [d for d in energy_dips
                        if d['depth'] * d['duration'] > 3.0]
            if big_dips:
                console.print(f"    {len(big_dips)} energy dips for structural breaks:")
                for d in big_dips:
                    console.print(f"      {d['start']:6.1f}s – {d['end']:6.1f}s  "
                                  f"(depth={d['depth']:.0%}, dur={d['duration']:.1f}s)")
                # Split vocal segments that fully contain a large dip
                new_segs: list[dict] = []
                for seg in vocal_segs:
                    splits_here = [d for d in big_dips
                                   if d['start'] > seg['start'] + 2.0
                                   and d['end'] < seg['end'] - 2.0]
                    if not splits_here:
                        new_segs.append(seg)
                        continue
                    # Sort splits by time and carve out
                    splits_here.sort(key=lambda d: d['start'])
                    cursor = seg['start']
                    for dip in splits_here:
                        if dip['start'] > cursor + 2.0:
                            new_segs.append({'start': cursor, 'end': dip['start']})
                        cursor = dip['end']
                    if seg['end'] > cursor + 2.0:
                        new_segs.append({'start': cursor, 'end': seg['end']})
                if len(new_segs) != len(vocal_segs):
                    vocal_segs = new_segs
                    console.print(f"    After energy-split: {len(vocal_segs)} vocal segments")
                    total_vocal_time = sum(s['end'] - s['start'] for s in vocal_segs)

    # ── 3. Identify gaps between vocal segments ──────────────────
    gaps: list[dict] = []
    for i in range(1, len(vocal_segs)):
        g_start = vocal_segs[i - 1]['end']
        g_end   = vocal_segs[i]['start']
        g_dur   = g_end - g_start
        if g_dur > 0.5:
            gaps.append({'start': g_start, 'end': g_end, 'duration': g_dur,
                         'seg_after': i})

    # Build "extended gaps" for instrumental matching: merge adjacent
    # gap–tiny_segment–gap sequences.  A short interstitial vocal segment
    # (< 5 s) between two gaps is likely a false positive (e.g. guitar
    # solo note that looks vocal).  Limit to absorbing at most ONE
    # interstitial segment to avoid chain-absorbing the whole song.
    ext_gaps: list[dict] = []
    skip_next = False
    for i in range(len(gaps)):
        if skip_next:
            skip_next = False
            continue
        g = gaps[i].copy()
        if i + 1 < len(gaps):
            seg_between_idx = gaps[i]['seg_after']
            seg_between = vocal_segs[seg_between_idx]
            seg_dur = seg_between['end'] - seg_between['start']
            if seg_dur < 5.0:
                next_gap = gaps[i + 1]
                g['end'] = next_gap['end']
                g['duration'] = g['end'] - g['start']
                g['seg_after'] = next_gap['seg_after']
                skip_next = True
        ext_gaps.append(g)

    console.print(f"    {len(gaps)} non-vocal gaps ({len(ext_gaps)} extended):")
    for g in ext_gaps:
        console.print(f"      {g['start']:6.1f}s – {g['end']:6.1f}s  "
                      f"({g['duration']:.1f}s)")

    # ── 4. Build ordered list of vocal blocks and instrumental markers ──
    blocks: list[dict] = []
    for sec in sections:
        if sec['is_instrumental']:
            blocks.append({'type': 'instrumental', 'sections': [sec],
                           'total_words': 0, 'all_lines': sec['lines']})
        else:
            if blocks and blocks[-1]['type'] == 'vocal':
                blocks[-1]['sections'].append(sec)
                blocks[-1]['total_words'] += sec['total_words']
                blocks[-1]['all_lines'].extend(sec['lines'])
            else:
                blocks.append({'type': 'vocal', 'sections': [sec],
                               'total_words': sec['total_words'],
                               'all_lines': list(sec['lines'])})

    console.print(f"    {len(blocks)} blocks: "
                  + ", ".join(
                      f"{'INST' if b['type'] == 'instrumental' else 'VOCAL'}"
                      f"({b['total_words']}w)" for b in blocks))

    # ── 5. Match instrumental blocks to the longest gaps ─────────
    # Use extended gaps for instrumental matching (they absorb small
    # false-positive vocal segments within solo/break regions).
    inst_indices = [i for i, b in enumerate(blocks) if b['type'] == 'instrumental']
    gap_claimed = [False] * len(ext_gaps)
    inst_gap_map: dict[int, dict] = {}   # block_idx → gap

    # For each instrumental block, estimate its expected proportional time
    total_vocal_words = sum(b['total_words'] for b in blocks)
    cum_words = 0
    for blk_idx, blk in enumerate(blocks):
        if blk['type'] == 'instrumental':
            expected_ratio = cum_words / total_vocal_words if total_vocal_words else 0.5
            expected_time = total_duration * expected_ratio

            # Find the best unclaimed gap: balance duration and proximity.
            # Use ratio scoring so a nearby shorter gap can beat a distant
            # longer one.
            best_gi = None
            best_score = -1.0
            for gi, gap in enumerate(ext_gaps):
                if gap_claimed[gi]:
                    continue
                dist = abs(gap['start'] - expected_time)
                score = gap['duration'] / (1.0 + dist / 15.0)
                if score > best_score:
                    best_score = score
                    best_gi = gi
            if best_gi is not None:
                inst_gap_map[blk_idx] = ext_gaps[best_gi]
                gap_claimed[best_gi] = True
        else:
            cum_words += blk['total_words']

    for blk_idx, gap in inst_gap_map.items():
        marker = blocks[blk_idx]['sections'][0]['marker']
        console.print(f"    Instrumental '{marker}' -> gap "
                      f"{gap['start']:.1f}s–{gap['end']:.1f}s")

    # ── 6. Determine split points in vocal segments ──────────────
    # The claimed extended gaps may span several original vocal segments
    # (those small false-positive segments that were absorbed).  We need
    # to exclude those absorbed segments from the assignment.
    inst_regions = sorted(
        (inst_gap_map[bi]['start'], inst_gap_map[bi]['end'])
        for bi in inst_gap_map
    )

    # Filter vocal segments: remove any that fall inside an instrumental region
    filtered_segs: list[dict] = []
    for seg in vocal_segs:
        inside_inst = False
        for ir_start, ir_end in inst_regions:
            if seg['start'] >= ir_start - 0.5 and seg['end'] <= ir_end + 0.5:
                inside_inst = True
                break
        if not inside_inst:
            filtered_segs.append(seg)

    if filtered_segs:
        vocal_segs_for_assignment = filtered_segs
    else:
        vocal_segs_for_assignment = vocal_segs   # fallback

    # Recalculate total vocal time after filtering
    total_vocal_time = sum(s['end'] - s['start'] for s in vocal_segs_for_assignment)
    console.print(f"    After filtering instrumentals: {len(vocal_segs_for_assignment)} "
                  f"vocal segs, {total_vocal_time:.1f}s")

    # Find the split index: for each instrumental region, find which
    # filtered segment comes right after it
    split_seg_indices: list[int] = []
    for ir_start, ir_end in inst_regions:
        for si, seg in enumerate(vocal_segs_for_assignment):
            if seg['start'] >= ir_end - 0.5:
                split_seg_indices.append(si)
                break

    # Build segment groups for each vocal block
    vocal_blocks = [b for b in blocks if b['type'] == 'vocal']
    seg_groups: list[list[dict]] = []
    prev_idx = 0
    for split_idx in sorted(set(split_seg_indices)):
        seg_groups.append(vocal_segs_for_assignment[prev_idx:split_idx])
        prev_idx = split_idx
    seg_groups.append(vocal_segs_for_assignment[prev_idx:])

    # If we have more vocal blocks than seg groups, merge the smallest
    # seg groups until counts match; if fewer, split the largest.
    while len(seg_groups) < len(vocal_blocks):
        # Split the group with the most total time
        biggest = max(range(len(seg_groups)),
                      key=lambda i: sum(s['end'] - s['start'] for s in seg_groups[i]))
        segs = seg_groups[biggest]
        mid = len(segs) // 2
        if mid > 0 and len(segs) > 1:
            seg_groups[biggest] = segs[:mid]
            seg_groups.insert(biggest + 1, segs[mid:])
        else:
            # Can't split further — leave as is
            break

    while len(seg_groups) > len(vocal_blocks) and len(vocal_blocks) > 0:
        # Merge the two smallest adjacent groups
        smallest = min(range(len(seg_groups) - 1),
                       key=lambda i: (sum(s['end'] - s['start'] for s in seg_groups[i])
                                     + sum(s['end'] - s['start'] for s in seg_groups[i + 1])))
        seg_groups[smallest] = seg_groups[smallest] + seg_groups[smallest + 1]
        del seg_groups[smallest + 1]

    console.print(f"    Vocal block -> segment-group assignment:")
    for vi, vblk in enumerate(vocal_blocks):
        if vi < len(seg_groups):
            grp = seg_groups[vi]
            grp_dur = sum(s['end'] - s['start'] for s in grp)
            console.print(f"      {vblk['total_words']:3d}w -> "
                          f"{len(grp)} segs, {grp_dur:.1f}s")

    # ── 7. Place lyrics using onset-driven timing ────────────────
    # Instead of proportional distribution (which doesn't match actual
    # singing rhythm), assign each lyric line to the next detected vocal
    # onset.  This gives karaoke-accurate timing where each line starts
    # when the singer actually starts singing it.
    subtitles: list[dict] = []

    for vi, vblk in enumerate(vocal_blocks):
        if vi >= len(seg_groups) or not seg_groups[vi]:
            continue

        grp = seg_groups[vi]
        grp_start = grp[0]['start']
        grp_end = grp[-1]['end']
        grp_total_time = sum(s['end'] - s['start'] for s in grp)

        # Get onsets within this block's segment group
        grp_onsets: list[float] = []
        for t in vocal_onsets:
            if t < grp_start - 0.3 or t > grp_end + 0.3:
                continue
            # Must be within an actual segment (not a gap between segments)
            for seg in grp:
                if seg['start'] - 0.2 <= t <= seg['end'] + 0.2:
                    grp_onsets.append(t)
                    break

        # Collect singing lines for this block
        singing_lines = [l for l in vblk['all_lines'] if l['type'] == 'lyric']
        n_singing = len(singing_lines)

        if n_singing == 0:
            continue

        console.print(f"      Block {vi}: {n_singing} lines, "
                      f"{len(grp_onsets)} onsets in {grp_total_time:.1f}s")

        # ── Match onsets to lines ──
        # If we have roughly enough onsets, use direct sequential assignment.
        # If too many onsets, select the N strongest that maintain order and
        # minimum spacing.  If too few, subdivide the available time.

        if len(grp_onsets) >= n_singing:
            if len(grp_onsets) == n_singing:
                chosen_onsets = grp_onsets
            else:
                # Too many onsets — select N from M using strength-aware
                # selection.  Get the onset strengths for onsets in this group.
                grp_onset_strengths: list[float] = []
                for t in grp_onsets:
                    best_strength = 0.0
                    for oi, rt in enumerate(raw_onsets):
                        if abs(rt - t) < 0.05 and oi < len(raw_strengths):
                            best_strength = raw_strengths[oi]
                            break
                    grp_onset_strengths.append(best_strength)

                # Score each onset: combine strength with distance to the
                # "expected" position (evenly spaced).  Strong onsets near
                # expected positions get top priority.
                expected_spacing = grp_total_time / n_singing

                # DP selection: pick N of M onsets minimizing total cost
                # where cost combines distance from expected and negative
                # strength (we want strong onsets).
                M_grp = len(grp_onsets)
                INF = float('inf')
                dp_sel   = [[INF] * M_grp for _ in range(n_singing)]
                back_sel = [[0]   * M_grp for _ in range(n_singing)]

                # Normalise strengths for scoring
                max_str = max(grp_onset_strengths) if grp_onset_strengths else 1.0
                if max_str == 0:
                    max_str = 1.0

                for j in range(M_grp):
                    expected_t = grp_start + expected_spacing * 0.1
                    dist_cost = abs(grp_onsets[j] - expected_t)
                    str_bonus = (grp_onset_strengths[j] / max_str) * 2.0
                    dp_sel[0][j] = dist_cost - str_bonus

                for i in range(1, n_singing):
                    run_min = INF
                    run_k = 0
                    expected_t = grp_start + expected_spacing * (i + 0.1)
                    for j in range(M_grp):
                        if j > 0 and dp_sel[i - 1][j - 1] < run_min:
                            run_min = dp_sel[i - 1][j - 1]
                            run_k = j - 1
                        if run_min < INF:
                            dist_cost = abs(grp_onsets[j] - expected_t)
                            str_bonus = (grp_onset_strengths[j] / max_str) * 2.0
                            cost = dist_cost - str_bonus + run_min
                            if cost < dp_sel[i][j]:
                                dp_sel[i][j] = cost
                                back_sel[i][j] = run_k

                best_j = int(np.argmin(dp_sel[n_singing - 1]))
                sel_assignment = [0] * n_singing
                sel_assignment[n_singing - 1] = best_j
                for i in range(n_singing - 2, -1, -1):
                    sel_assignment[i] = back_sel[i + 1][sel_assignment[i + 1]]

                chosen_onsets = [grp_onsets[sel_assignment[i]]
                                 for i in range(n_singing)]
        else:
            # Fewer onsets than lines — use available onsets and subdivide
            # the time between consecutive onsets for remaining lines
            chosen_onsets = list(grp_onsets)
            # Add evenly-spaced synthetic onsets to fill gaps
            while len(chosen_onsets) < n_singing:
                # Find the largest gap between consecutive onsets (or
                # between start/end and first/last onset)
                points = [grp_start] + chosen_onsets + [grp_end]
                max_gap = 0
                max_gap_idx = 0
                for gi in range(len(points) - 1):
                    gap = points[gi + 1] - points[gi]
                    if gap > max_gap:
                        max_gap = gap
                        max_gap_idx = gi
                # Insert midpoint of largest gap
                mid = (points[max_gap_idx] + points[max_gap_idx + 1]) / 2
                chosen_onsets.append(mid)
                chosen_onsets.sort()

        # ── Enforce minimum spacing between chosen onsets ──
        # Merge any onsets closer than min_onset_gap — otherwise lines get
        # squished to unreadable durations.
        filtered_chosen: list[float] = [chosen_onsets[0]]
        for ci in range(1, len(chosen_onsets)):
            if chosen_onsets[ci] - filtered_chosen[-1] >= min_onset_gap:
                filtered_chosen.append(chosen_onsets[ci])
            # else: skip this too-close onset
        chosen_onsets = filtered_chosen

        # If filtering removed too many, we may have fewer onsets than lines.
        # Re-fill by splitting largest gaps.
        while len(chosen_onsets) < n_singing:
            points = [grp_start] + chosen_onsets + [grp_end]
            max_gap = 0
            max_gap_idx = 0
            for gi in range(len(points) - 1):
                gap = points[gi + 1] - points[gi]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = gi
            if max_gap < min_onset_gap * 0.8:
                break  # Can't split further without violating spacing
            mid = (points[max_gap_idx] + points[max_gap_idx + 1]) / 2
            chosen_onsets.append(mid)
            chosen_onsets.sort()

        console.print(f"        -> {len(chosen_onsets)} chosen onsets for "
                      f"{n_singing} lines")

        # ── Assign lines to chosen onsets ──
        onset_idx = 0
        all_items = list(vblk['all_lines'])
        
        # Collect all lyric lines first
        lyric_lines = [l for l in all_items if l['type'] == 'lyric']
        
        for line_idx, line in enumerate(lyric_lines):
            # Determine start time
            if onset_idx < len(chosen_onsets):
                start = chosen_onsets[onset_idx] - pre_roll
                start = max(0, start)
                onset_idx += 1
                using_onset = True
            else:
                # No more onsets - space remaining lines evenly to end
                remaining_lines = len(lyric_lines) - line_idx
                if subtitles:
                    last_end = subtitles[-1]['end']
                else:
                    last_end = grp_start
                available_time = grp_end - last_end
                spacing = max(2.0, available_time / max(1, remaining_lines))
                start = last_end + 0.1  # Small gap after previous
                using_onset = False

            # End time calculation
            if using_onset and onset_idx < len(chosen_onsets):
                # There's another onset coming
                next_start = chosen_onsets[onset_idx] - pre_roll
                end = next_start - 0.05
            elif not using_onset:
                # Evenly spaced - use spacing for end
                end = start + spacing - 0.1
            else:
                # Last line or no more onsets: extend to end of segment group
                end = grp_end

            # Clamp duration: minimum for readability
            dur = end - start
            dur = max(min_duration, min(dur, 8.0))
            end = start + dur

            # Don't exceed segment boundaries — but allow a small
            # overshoot (up to 0.5 s) so lines near segment edges still
            # meet the 1.5 s minimum display time.
            for seg in grp:
                if seg['start'] - 0.1 <= start <= seg['end'] + 0.1:
                    hard_limit = seg['end'] + 0.5   # slight grace past boundary
                    end = min(end, hard_limit)
                    # Re-enforce minimum after clamping
                    if end - start < 1.5:
                        end = start + 1.5
                    end = min(end, hard_limit)
                    break

            if start >= end:
                continue

            subtitles.append({
                'index': len(subtitles) + 1,
                'start': round(start, 3),
                'end':   round(end, 3),
                'text':  line['text'],
            })

    # ── 8. Instrumental markers skipped ──────────────────────────
    # Bracketed markers like [Guitar Solo], [Verse] etc. are used
    # internally for section timing but NOT added as subtitle text.

    # Sort by start time and re-index
    subtitles.sort(key=lambda s: s['start'])
    for i, sub in enumerate(subtitles):
        sub['index'] = i + 1

    # Don't exceed total duration
    for sub in subtitles:
        sub['end'] = min(sub['end'], total_duration - 0.1)

    # ── Post-pass: fix short lines ───────────────────────────────
    # If a subtitle is shorter than min_duration, pull its start earlier
    # (up to the previous subtitle's end + 0.05 s gap).  Appearing
    # slightly before the vocal onset is much better UX than a
    # subtitle that flashes too quickly to read.
    for i, sub in enumerate(subtitles):
        dur = sub['end'] - sub['start']
        if dur >= min_duration - 0.01:       # tolerance for float rounding
            continue
        needed = min_duration - dur
        # How far back can we go?
        earliest = subtitles[i - 1]['end'] + 0.05 if i > 0 else 0.0
        new_start = max(earliest, sub['start'] - needed)
        sub['start'] = round(new_start, 3)
        # If still too short after pulling start, extend end a bit
        if sub['end'] - sub['start'] < min_duration - 0.01:
            sub['end'] = round(sub['start'] + min_duration + 0.01, 3)
        # Avoid overlapping next subtitle
        if i + 1 < len(subtitles) and sub['end'] > subtitles[i + 1]['start'] - 0.05:
            sub['end'] = round(subtitles[i + 1]['start'] - 0.05, 3)

    console.print(f"    Generated {len(subtitles)} subtitle cues")
    if subtitles:
        console.print(f"    First cue at {subtitles[0]['start']:.1f}s, "
                      f"last ends at {subtitles[-1]['end']:.1f}s")

    return subtitles


def _proportional_fallback(sections: list[dict], total_duration: float) -> list[dict]:
    """Simple proportional fallback if energy analysis fails."""
    subtitles = []
    cursor = total_duration * 0.03
    available = total_duration * 0.92
    total_words = sum(s.get('total_words', 0) for s in sections) or 1
    for sec in sections:
        for line in sec.get('lines', []):
            if line['type'] == 'pause':
                cursor += 0.5
                continue
            dur = max(1.5, (line['word_count'] / total_words) * available)
            dur = min(dur, 6.0)
            subtitles.append({
                'index': len(subtitles) + 1,
                'start': round(cursor, 3),
                'end':   round(cursor + dur, 3),
                'text':  line['text'],
            })
            cursor += dur
    return subtitles


# ── Energy-based timing refinement ───────────────────────────────

def _read_wav_samples(wav_path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return normalised mono float samples and sample rate."""
    wf = wave.open(wav_path, 'r')
    rate = wf.getframerate()
    channels = wf.getnchannels()
    width = wf.getsampwidth()
    n_frames = wf.getnframes()
    raw = wf.readframes(n_frames)
    wf.close()

    if width == 2:
        fmt = f'<{n_frames * channels}h'
        samples = struct.unpack(fmt, raw)
        norm = 32768.0
    elif width == 4:
        fmt = f'<{n_frames * channels}i'
        samples = struct.unpack(fmt, raw)
        norm = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {width}")

    # Mix to mono and normalise to [-1, 1]  (vectorised with numpy)
    arr = np.array(samples, dtype=np.float64)
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    mono = arr / norm

    return mono, rate


# ── Spectral Voice Activity Detection (VAD) ─────────────────────

def _compute_vocal_score(wav_path: str) -> tuple[np.ndarray, float, int]:
    """
    Compute per-frame vocal presence score using spectral band analysis.

    Uses three complementary features to distinguish singing from instruments:
      1. Vocal-band energy ratio  (300–4000 Hz vs full spectrum)
      2. Spectral flux in the vocal range (frame-to-frame change — singing
         has rapid spectral variation due to changing syllables)
      3. Vocal-band energy modulation (coefficient of variation over a 2 s
         sliding window — singing produces syllabic amplitude bursts at
         ~3–7 Hz, whereas sustained guitar is steadier)

    Returns (scores, hop_sec, sample_rate).
    """
    mono, rate = _read_wav_samples(wav_path)
    audio = np.asarray(mono, dtype=np.float64)

    frame_size = 2048                       # ~46 ms at 44100 Hz
    hop_size = 512                          # ~11.6 ms
    hop_sec = hop_size / rate
    n_frames = max(0, (len(audio) - frame_size) // hop_size)

    if n_frames == 0:
        return np.array([]), hop_sec, rate

    window = np.hanning(frame_size)
    freq_res = rate / frame_size

    def fbin(hz):
        return max(0, min(int(hz / freq_res), frame_size // 2))

    # ---- frequency-band edges ----
    FULL_LO,   FULL_HI   = fbin(50),   fbin(10000)
    BASS_LO,   BASS_HI   = fbin(50),   fbin(300)
    VOCAL_LO,  VOCAL_HI  = fbin(300),  fbin(4000)
    HMID_LO,   HMID_HI   = fbin(1500), fbin(4000)

    vocal_ratio    = np.zeros(n_frames)
    spectral_flux  = np.zeros(n_frames)
    vocal_band_amp = np.zeros(n_frames)          # for modulation feature
    prev_vocal_spec = None

    for i in range(n_frames):
        s = i * hop_size
        frame = audio[s : s + frame_size] * window
        spec = np.abs(np.fft.rfft(frame))
        power = spec ** 2

        total_e  = np.sum(power[FULL_LO:FULL_HI]) + 1e-10
        vocal_e  = np.sum(power[VOCAL_LO:VOCAL_HI])
        hmid_e   = np.sum(power[HMID_LO:HMID_HI])
        bass_e   = np.sum(power[BASS_LO:BASS_HI]) + 1e-10

        # Feature 1: vocal-band ratio
        vocal_ratio[i] = vocal_e / total_e

        # Feature 2: spectral flux in vocal range
        vocal_spec = spec[VOCAL_LO:VOCAL_HI]
        if prev_vocal_spec is not None:
            diff = vocal_spec - prev_vocal_spec
            spectral_flux[i] = np.sum(np.maximum(diff, 0)) / (np.sum(vocal_spec) + 1e-10)
        prev_vocal_spec = vocal_spec.copy()

        # Amplitude envelope of vocal band (for modulation score)
        vocal_band_amp[i] = np.sqrt(vocal_e)

    # Feature 3: syllabic modulation — coefficient of variation of
    # vocal-band amplitude over a 2 s sliding window (vectorised).
    mod_win = max(1, int(2.0 / hop_sec))
    # Sliding-window mean and variance via cumsum trick
    cs  = np.cumsum(np.concatenate([[0], vocal_band_amp]))
    cs2 = np.cumsum(np.concatenate([[0], vocal_band_amp ** 2]))
    half = mod_win // 2
    lo_idx = np.clip(np.arange(n_frames) - half, 0, n_frames)
    hi_idx = np.clip(np.arange(n_frames) + half + 1, 0, n_frames)
    counts = hi_idx - lo_idx
    win_mean = (cs[hi_idx] - cs[lo_idx]) / counts
    win_var  = (cs2[hi_idx] - cs2[lo_idx]) / counts - win_mean ** 2
    win_std  = np.sqrt(np.maximum(win_var, 0))
    energy_cv = np.where(win_mean > 1e-8, win_std / win_mean, 0.0)

    # ---- smooth each feature (500 ms) ----
    k = max(1, int(0.5 / hop_sec))
    kernel = np.ones(k) / k
    vocal_ratio   = np.convolve(vocal_ratio,   kernel, mode='same')
    spectral_flux = np.convolve(spectral_flux, kernel, mode='same')
    energy_cv     = np.convolve(energy_cv,     kernel, mode='same')

    # ---- normalise to [0, 1] ----
    def norm01(x):
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        return np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x)

    vr = norm01(vocal_ratio)
    sf = norm01(spectral_flux)
    cv = norm01(energy_cv)

    # ---- combine ----
    score = vr * 0.30 + sf * 0.40 + cv * 0.30

    # Broad smoothing for phrase-level activity
    k2 = max(1, int(0.5 / hop_sec))
    kernel2 = np.ones(k2) / k2
    score = np.convolve(score, kernel2, mode='same')

    return score, hop_sec, rate


def detect_vocal_segments(wav_path: str,
                          min_vocal_sec: float = 2.0,
                          min_gap_sec: float = 1.5) -> tuple[list[dict], np.ndarray, float]:
    """
    Detect time regions where vocals are actively present.

    Returns
    -------
    segments : list[dict]   – [{'start': float, 'end': float}, ...]
    score    : np.ndarray   – raw per-frame vocal activity score
    hop_sec  : float        – time resolution of *score*
    """
    score, hop_sec, rate = _compute_vocal_score(wav_path)

    if len(score) == 0:
        return [], score, hop_sec

    # ── Otsu adaptive threshold ──────────────────────────────────
    hist, bin_edges = np.histogram(score, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    sum_total = np.sum(hist * bin_centers)

    best_threshold = float(np.median(score))
    best_var = 0.0
    weight_bg = 0.0
    sum_bg = 0.0

    for i in range(len(hist)):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_threshold = float(bin_centers[i])

    is_vocal = score > best_threshold

    # ── Secondary local-threshold pass (vectorised) ──────────────
    # The Otsu threshold is global and may miss quieter vocal passages
    # (e.g. harmonised sections, clean-sung bridges).  A local adaptive
    # threshold catches vocals that are locally prominent even when they
    # don't cross the global threshold.
    block_size = max(1, int(5.0 / hop_sec))       # 5-second blocks
    n_blocks = (len(score) + block_size - 1) // block_size
    padded = np.pad(score, (0, n_blocks * block_size - len(score)),
                    constant_values=0)
    blocks = padded[:n_blocks * block_size].reshape(n_blocks, block_size)
    block_maxes = np.max(blocks, axis=1)

    # Expand to ±1 block (≈15 s context)
    expanded = block_maxes.copy()
    if n_blocks > 1:
        expanded[:-1] = np.maximum(expanded[:-1], block_maxes[1:])
        expanded[1:]  = np.maximum(expanded[1:],  block_maxes[:-1])

    local_max_pf = np.repeat(expanded, block_size)[:len(score)]
    is_vocal = is_vocal | (score > local_max_pf * 0.55)

    # ── Extract raw segments ─────────────────────────────────────
    segments: list[dict] = []
    in_seg = False
    seg_start = 0.0

    for i in range(len(is_vocal)):
        if is_vocal[i] and not in_seg:
            in_seg = True
            seg_start = i * hop_sec
        elif not is_vocal[i] and in_seg:
            in_seg = False
            segments.append({'start': seg_start, 'end': i * hop_sec})
    if in_seg:
        segments.append({'start': seg_start, 'end': len(score) * hop_sec})

    # ── Merge segments closer than min_gap_sec ───────────────────
    merged: list[dict] = []
    for seg in segments:
        if merged and seg['start'] - merged[-1]['end'] < min_gap_sec:
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg.copy())

    # ── Remove very short segments ───────────────────────────────
    merged = [s for s in merged if s['end'] - s['start'] >= min_vocal_sec]

    return merged, score, hop_sec


def _map_vocal_timeline_to_audio(vt_pos: float, vocal_segs: list[dict]) -> float:
    """Map a position in the concatenated vocal-only timeline to actual audio time."""
    cumulative = 0.0
    for seg in vocal_segs:
        seg_dur = seg['end'] - seg['start']
        if cumulative + seg_dur > vt_pos:
            return seg['start'] + (vt_pos - cumulative)
        cumulative += seg_dur
    return vocal_segs[-1]['end'] if vocal_segs else 0.0


def compute_energy_profile(wav_path: str, window_sec: float = 0.1,
                           smooth_sec: float = 1.0) -> tuple[list[float], float]:
    """
    Compute smoothed energy profile of the audio.
    Returns (energy_values, window_sec) where each value is the RMS energy
    for that time window.
    """
    mono, rate = _read_wav_samples(wav_path)
    samples = np.asarray(mono, dtype=np.float64)
    window_samples = int(rate * window_sec)
    n_windows = len(samples) // window_samples

    if n_windows == 0:
        return [], window_sec

    # Vectorised RMS per window
    trimmed = samples[:n_windows * window_samples].reshape(n_windows, window_samples)
    energies = np.sqrt(np.mean(trimmed ** 2, axis=1))

    # Smooth with uniform kernel
    smooth_windows = max(1, int(smooth_sec / window_sec))
    kernel = np.ones(smooth_windows) / smooth_windows
    smoothed = np.convolve(energies, kernel, mode='same')

    return smoothed.tolist(), window_sec


def detect_audio_boundaries(energy_profile: list[float], window_sec: float,
                            total_duration: float) -> tuple[float, float]:
    """
    Detect when the audio actually starts and ends (not silence).
    Returns (audio_start_time, audio_end_time).
    """
    if not energy_profile:
        return 0.0, total_duration

    max_e = max(energy_profile)
    threshold = max_e * 0.05  # 5% of peak = "audio has started"

    # Find first window above threshold
    audio_start = 0.0
    for i, e in enumerate(energy_profile):
        if e > threshold:
            audio_start = max(0, i * window_sec - 0.5)
            break

    # Find last window above threshold
    audio_end = total_duration
    for i in range(len(energy_profile) - 1, -1, -1):
        if energy_profile[i] > threshold:
            audio_end = min(total_duration, (i + 1) * window_sec + 0.5)
            break

    return audio_start, audio_end


def detect_energy_dips(energy_profile: list[float], window_sec: float,
                       total_duration: float, min_dip_sec: float = 2.0) -> list[dict]:
    """
    Find significant energy dips — moments where the music drops substantially.
    These correspond to section transitions, instrumental breaks, solos, etc.

    Returns list of dicts: [{'time': float, 'depth': float, 'duration': float}, ...]
    sorted by time. 'time' is the centre of the dip, 'depth' is how much the
    energy drops relative to the local context (0..1), and 'duration' is the
    dip width in seconds.
    """
    if len(energy_profile) < 10:
        return []

    # Compute a running "local max" envelope over ~10s windows
    context_windows = max(1, int(10.0 / window_sec))
    local_max = []
    for i in range(len(energy_profile)):
        s = max(0, i - context_windows)
        e = min(len(energy_profile), i + context_windows + 1)
        local_max.append(max(energy_profile[s:e]))

    # Find dips: regions where energy drops below 40% of local context
    dips = []
    in_dip = False
    dip_start = 0
    dip_min_ratio = 1.0

    for i in range(len(energy_profile)):
        if local_max[i] > 0:
            ratio = energy_profile[i] / local_max[i]
        else:
            ratio = 1.0

        if ratio < 0.40:
            if not in_dip:
                in_dip = True
                dip_start = i
                dip_min_ratio = ratio
            else:
                dip_min_ratio = min(dip_min_ratio, ratio)
        else:
            if in_dip:
                in_dip = False
                dip_end = i
                dip_duration = (dip_end - dip_start) * window_sec
                if dip_duration >= min_dip_sec:
                    dip_centre = ((dip_start + dip_end) / 2) * window_sec
                    dips.append({
                        'time': dip_centre,
                        'start': dip_start * window_sec,
                        'end': dip_end * window_sec,
                        'depth': 1.0 - dip_min_ratio,
                        'duration': dip_duration,
                    })

    return dips


def align_lyrics_with_energy(lyrics_lines: list[dict], wav_path: str,
                             total_duration: float) -> list[dict]:
    """
    Energy-informed proportional subtitle timing.

    Uses energy analysis to find *structural* boundaries only:
      - Intro dip  → vocals start AFTER it
      - Outro dip  → vocals end BEFORE it
      - Mid-song dips → dead zones (guitar solos, breaks) where no lyrics appear

    All lyrics are then distributed proportionally (by word-count weight)
    across the remaining vocal windows — the same approach the proportional
    timer uses, but with accurate start/end/break positions.
    """
    console.print("    Analyzing audio energy...")
    energy_profile, window_sec = compute_energy_profile(wav_path)

    if not energy_profile:
        return []

    # ── 1. Find energy dips ──────────────────────────────────────
    dips = detect_energy_dips(energy_profile, window_sec, total_duration)
    console.print(f"    Found {len(dips)} energy dips")
    for d in dips:
        console.print(f"      {d['start']:.1f}s–{d['end']:.1f}s  "
                       f"depth={d['depth']:.0%}  dur={d['duration']:.1f}s")

    # ── 2. Classify dips as intro / structural / outro ───────────
    vocal_start = total_duration * 0.03        # conservative default
    vocal_end   = total_duration - 1.0
    dead_zones: list[tuple[float, float]] = []  # (start, end)

    for dip in dips:
        if dip['start'] < total_duration * 0.10:
            # Dip within first 10% → intro.  Vocals begin after it.
            vocal_start = max(vocal_start, dip['end'])
        elif dip['end'] > total_duration * 0.93:
            # Dip in last 7% → outro.  Vocals end before it.
            vocal_end = min(vocal_end, dip['start'])
        else:
            # Mid-song structural break
            dead_zones.append((dip['start'], dip['end']))

    dead_zones.sort()
    console.print(f"    Vocal window: {vocal_start:.1f}s – {vocal_end:.1f}s")
    if dead_zones:
        console.print(f"    Dead zones: {dead_zones}")

    # ── 3. Build available vocal segments ────────────────────────
    segments: list[tuple[float, float]] = []
    cursor = vocal_start
    for dz_start, dz_end in dead_zones:
        if dz_start > cursor + 0.5:
            segments.append((cursor, dz_start))
        cursor = max(cursor, dz_end)
    if vocal_end > cursor + 0.5:
        segments.append((cursor, vocal_end))
    if not segments:
        segments = [(vocal_start, vocal_end)]

    total_avail = sum(e - s for s, e in segments)
    console.print(f"    {len(segments)} vocal segments, {total_avail:.1f}s available")

    # ── 4. Build weighted display items from lyrics ──────────────
    items: list[dict] = []
    for line in lyrics_lines:
        if line.get('is_blank'):
            # Blank lines → small pause (advances time but shows nothing)
            items.append({'text': '', 'type': 'pause', 'weight': 0.3})
        elif line.get('is_direction'):
            items.append({'text': line['text'], 'type': 'direction', 'weight': 0.5})
        else:
            wc = len(line['text'].split())
            items.append({'text': line['text'], 'type': 'lyric',
                          'weight': max(1.0, wc * 0.35)})

    if not items:
        return []

    total_weight = sum(it['weight'] for it in items)
    time_per_weight = total_avail / total_weight

    # ── 5. First pass: compute raw durations with min/max clamps ─
    durations: list[float] = []
    for it in items:
        raw = it['weight'] * time_per_weight
        if it['type'] == 'pause':
            dur = raw                                   # no clamp on pauses
        elif it['type'] == 'direction':
            dur = max(1.0, min(raw, 1.5))
        else:
            dur = max(1.5, min(raw, 6.0))
        durations.append(dur)

    # Redistribute any surplus from clamped items
    used = sum(durations)
    surplus = total_avail - used
    if surplus > 1.0:
        # Give extra to non-clamped lyric items proportionally
        expandable = [(i, items[i]['weight'])
                      for i in range(len(items))
                      if items[i]['type'] == 'lyric' and durations[i] < 6.0]
        if expandable:
            exp_weight = sum(w for _, w in expandable)
            for idx, w in expandable:
                extra = surplus * (w / exp_weight)
                durations[idx] = min(6.0, durations[idx] + extra)

    # ── 6. Place items sequentially across vocal segments ────────
    subtitles: list[dict] = []
    seg_idx = 0
    seg_cursor = segments[0][0]

    def advance_to_next_segment():
        nonlocal seg_idx, seg_cursor
        if seg_idx + 1 < len(segments):
            seg_idx += 1
            seg_cursor = segments[seg_idx][0]

    for i, it in enumerate(items):
        dur = durations[i]

        # Pauses just advance time
        if it['type'] == 'pause':
            seg_cursor += dur
            # If we've overrun the segment, spill into the next one
            while seg_idx < len(segments) - 1 and seg_cursor >= segments[seg_idx][1]:
                overshoot = seg_cursor - segments[seg_idx][1]
                advance_to_next_segment()
                seg_cursor = segments[seg_idx][0] + overshoot
            continue

        # If we're past the end of this segment, move to next
        if seg_cursor >= segments[seg_idx][1] - 0.3:
            advance_to_next_segment()

        # Would this item cross a segment boundary?
        item_end = seg_cursor + dur
        if item_end > segments[seg_idx][1]:
            remaining = segments[seg_idx][1] - seg_cursor
            if remaining >= 1.2:
                # Enough room — just truncate to fit
                item_end = segments[seg_idx][1]
            else:
                # Not enough room — jump to next segment
                advance_to_next_segment()
                item_end = segments[seg_idx][0] + dur
                seg_cursor = segments[seg_idx][0]
                if item_end > segments[seg_idx][1]:
                    item_end = segments[seg_idx][1]

        if seg_cursor >= item_end:
            continue

        subtitles.append({
            'index': len(subtitles) + 1,
            'start': round(seg_cursor, 3),
            'end':   round(item_end, 3),
            'text':  it['text'],
        })
        seg_cursor = item_end

    return subtitles


# ── Phrase-onset detection and subtitle snapping ─────────────────

def detect_phrase_onsets(wav_path: str) -> tuple[list[float], list[float]]:
    """
    Detect phrase-level vocal onset times from a WAV file.

    Uses **vocal-band energy** (300-4000 Hz) instead of broadband energy,
    so drum hits and bass transients are ignored.  Returns both onset times
    and their strengths (derivative peak magnitude) so that the alignment
    stage can select the N most prominent vocal entries.

    Returns (onset_times, onset_strengths), both sorted by time.
    """
    mono, rate = _read_wav_samples(wav_path)
    audio = np.asarray(mono, dtype=np.float64)

    # FFT-based vocal-band energy extraction
    frame_size = 2048                       # ~46 ms at 44100 Hz
    hop_size = 512                          # ~11.6 ms
    hop_sec = hop_size / rate
    n_frames = max(0, (len(audio) - frame_size) // hop_size)

    if n_frames == 0:
        return [], []

    window = np.hanning(frame_size)
    freq_res = rate / frame_size
    vlo = max(0, int(300 / freq_res))
    vhi = min(frame_size // 2, int(4000 / freq_res))

    vocal_energy = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * hop_size
        frame = audio[s : s + frame_size] * window
        spec = np.abs(np.fft.rfft(frame))
        vocal_energy[i] = np.sqrt(np.sum(spec[vlo:vhi] ** 2) + 1e-10)

    # 60 ms smoothing — tight enough for metal vocal attacks while
    # still filtering sub-beat noise.  Was 100 ms but added ~50 ms latency.
    k = max(1, int(0.06 / hop_sec))
    kernel = np.ones(k) / k
    smooth = np.convolve(vocal_energy, kernel, mode='same')

    # Positive derivative with 50 ms lookback — catches fast vocal attacks
    # without adding excessive delay.  Was 80 ms.
    lb = max(1, int(0.05 / hop_sec))
    deriv = np.zeros(len(smooth))
    deriv[lb:] = smooth[lb:] - smooth[:-lb]
    deriv = np.clip(deriv, 0, None)

    pos_vals = deriv[deriv > 0]
    if len(pos_vals) == 0:
        return [], []

    # Adaptive threshold: top 32% of positive derivative values
    threshold = float(np.percentile(pos_vals, 68))

    # Peak-pick with >= 1.2 s minimum gap — line-level detection.
    # Metal vocal lines are typically 1.5–4 s apart.  0.6 s catches
    # individual syllables/beats; 2.0 s was too coarse.  1.2 s is the
    # sweet spot for line-level entries (≈ 50–80 onsets per song).
    min_gap_idx = max(1, int(1.2 / hop_sec))
    onsets: list[float] = []
    strengths: list[float] = []
    i = 0
    while i < len(deriv):
        if deriv[i] > threshold:
            rise_start = i
            j = i + 1
            while j < len(deriv) and deriv[j] > threshold * 0.3:
                j += 1
            peak_idx = rise_start + int(np.argmax(deriv[rise_start:j]))
            peak_strength = float(deriv[peak_idx])
            # Use rise_start as the onset time — this is when the vocal
            # energy actually starts climbing.  The midpoint or peak is
            # too late (the singer has already been singing for 100+ ms).
            # Subtract ~80 ms to compensate for analysis latency (30 ms
            # from smoothing group delay + 50 ms from derivative lookback).
            onsets.append(max(0, rise_start * hop_sec - 0.08))
            strengths.append(peak_strength)
            i = max(j, rise_start + min_gap_idx)
        else:
            i += 1

    return onsets, strengths


def align_subtitles_to_audio(subtitles: list[dict], wav_path: str,
                              total_duration: float,
                              use_whisper: bool = True,
                              whisper_model: str = "base",
                              language: str | None = None,
                              vocal_start: float = 0.0) -> list[dict]:
    """
    Align subtitle timing to the actual audio using voice recognition.
    
    Uses Whisper AI for accurate word-level transcription and alignment.
    Falls back to energy-based onset detection if Whisper is unavailable.
    
    Parameters
    ----------
    subtitles : list[dict]
        Subtitles with 'text', 'start', 'end' keys
    wav_path : str
        Path to WAV audio file
    total_duration : float
        Total audio duration in seconds
    use_whisper : bool
        Whether to try Whisper first (default True)
    whisper_model : str
        Whisper model size: 'tiny', 'base', 'small', 'medium', 'large-v3'
    language : str, optional
        Language code for transcription (auto-detected if None)
    vocal_start : float
        User-specified time when vocals start (anchors alignment)
    
    Returns
    -------
    list[dict]
        Subtitles with adjusted start/end times
    """
    # Try Whisper first
    if use_whisper and _HAS_WHISPER:
        try:
            console.print("[cyan]Using Whisper for voice recognition...[/cyan]")
            whisper_words = transcribe_with_whisper(
                wav_path, 
                language=language,
                model_size=whisper_model
            )
            if whisper_words:
                result = align_lyrics_to_whisper(subtitles, whisper_words, total_duration, vocal_start)
                console.print("[green]Whisper alignment complete[/green]")
                return result
            else:
                console.print("[yellow]Whisper returned no words, falling back...[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Whisper failed: {e}, falling back to energy-based...[/yellow]")
    
    # Fallback: energy-based onset detection
    console.print("[cyan]Using energy-based onset detection...[/cyan]")
    return _align_subtitles_energy_based(subtitles, wav_path, total_duration)


def _align_subtitles_energy_based(subtitles: list[dict], wav_path: str,
                                   total_duration: float) -> list[dict]:
    """
    Fallback alignment using energy-based onset detection.
    
    Fine-tune subtitle start times by snapping to the nearest detected
    vocal onset using dynamic-programming optimal assignment.
    """
    # ── Detect vocal segments and onsets ──
    vocal_segs, _, _ = detect_vocal_segments(wav_path)
    raw_onsets, raw_strengths = detect_phrase_onsets(wav_path)

    if not vocal_segs:
        return subtitles

    # Filter onsets to vocal segments
    onsets: list[float] = []
    for t in raw_onsets:
        for seg in vocal_segs:
            if seg['start'] - 0.3 <= t <= seg['end'] + 0.3:
                onsets.append(t)
                break

    # Inject segment-start onsets (first phrase may already be loud)
    for seg in vocal_segs:
        t0 = seg['start'] + 0.1
        if all(abs(t0 - o) > 0.8 for o in onsets):
            onsets.append(t0)
    onsets.sort()

    console.print(f"    Onsets: {len(raw_onsets)} raw -> {len(onsets)} vocal-filtered")

    adjusted = [sub.copy() for sub in subtitles]

    # ── Identify singing vs direction lines ──
    def _is_direction(text: str) -> bool:
        t = text.strip()
        return ((t.startswith('[') and t.endswith(']'))
                or (t.startswith('(') and t.endswith('):'))
                or (t.startswith('(') and t.endswith(')')))

    singing_indices = [i for i, s in enumerate(adjusted)
                       if not _is_direction(s['text'])]
    N = len(singing_indices)
    M = len(onsets)

    if N == 0 or M == 0:
        return adjusted

    # Use the onset-driven positions directly as expected positions
    expected = [adjusted[si]['start'] for si in singing_indices]

    # ── DP: find optimal assignment of N lines to N of M onsets ──
    # dp[i][j] = min total cost to assign lines 0..i  with line i → onset j
    # Transition: dp[i][j] = cost(i,j) + min_{k < j} dp[i-1][k]
    # We track the running minimum to achieve O(N·M).
    MAX_SNAP = 4.0  # max seconds to snap away — tighter since initial is onset-driven
    INF = float('inf')

    dp   = [[INF] * M for _ in range(N)]
    back = [[0]   * M for _ in range(N)]

    # Base case: first singing line
    for j in range(M):
        dp[0][j] = abs(expected[0] - onsets[j])

    # Fill DP table
    for i in range(1, N):
        run_min = INF
        run_k = 0
        for j in range(M):
            # Update running minimum from previous row (columns < j)
            if j > 0 and dp[i - 1][j - 1] < run_min:
                run_min = dp[i - 1][j - 1]
                run_k = j - 1
            if run_min < INF:
                cost = abs(expected[i] - onsets[j]) + run_min
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    back[i][j] = run_k

    # Backtrack to recover optimal assignment
    best_j = int(np.argmin(dp[N - 1]))
    assignment = [0] * N
    assignment[N - 1] = best_j
    for i in range(N - 2, -1, -1):
        assignment[i] = back[i + 1][assignment[i + 1]]

    # ── Apply onset times to subtitle start positions ──
    # Subtract PRE_ROLL so the subtitle appears slightly before the singer
    # starts — standard for karaoke / broadcast subtitling.
    snapped_count = 0
    for i, si in enumerate(singing_indices):
        onset_t = onsets[assignment[i]]
        distance = abs(onset_t - expected[i])
        if distance <= MAX_SNAP:
            adjusted[si]['start'] = round(max(0, onset_t - SUBTITLE_PRE_ROLL), 3)
            snapped_count += 1
        # else: keep proportional position

    console.print(f"    DP matched {snapped_count}/{N} singing lines "
                  f"(within {MAX_SNAP}s of expected)")

    # ── Recompute end times ──
    # Each singing line ends just before the next subtitle starts.
    for idx in range(len(singing_indices) - 1):
        si      = singing_indices[idx]
        si_next = singing_indices[idx + 1]
        # If there are direction cues between the two singing lines,
        # find the earliest start among them.
        next_start = adjusted[si_next]['start']
        for k in range(si + 1, si_next):
            if adjusted[k]['start'] < next_start:
                next_start = adjusted[k]['start']
        new_end = next_start - 0.05
        # Keep original proportional duration as a cap so lines don't
        # become absurdly long during instrumental gaps.
        orig_dur = subtitles[si]['end'] - subtitles[si]['start']
        max_end = adjusted[si]['start'] + max(orig_dur * 1.3, 3.0)
        adjusted[si]['end'] = round(min(new_end, max_end), 3)
        # Enforce minimum 1.5 s
        if adjusted[si]['end'] < adjusted[si]['start'] + 1.5:
            adjusted[si]['end'] = round(adjusted[si]['start'] + 1.5, 3)

    # Last singing line: use capped original duration
    last_si = singing_indices[-1]
    orig_dur = subtitles[last_si]['end'] - subtitles[last_si]['start']
    adjusted[last_si]['end'] = round(
        min(adjusted[last_si]['start'] + max(orig_dur, 1.5),
            total_duration - 0.1), 3)

    # ── Reposition direction markers just before their next singing line ──
    for i in range(len(adjusted)):
        if _is_direction(adjusted[i]['text']):
            dur = adjusted[i]['end'] - adjusted[i]['start']
            for j in range(i + 1, len(adjusted)):
                if not _is_direction(adjusted[j]['text']):
                    adjusted[i]['end']   = round(adjusted[j]['start'] - 0.05, 3)
                    adjusted[i]['start'] = round(max(0, adjusted[i]['end'] - dur), 3)
                    break

    # ── Fix overlaps ──
    for i in range(1, len(adjusted)):
        if adjusted[i]['start'] < adjusted[i - 1]['end']:
            adjusted[i - 1]['end'] = round(adjusted[i]['start'] - 0.02, 3)
            if adjusted[i - 1]['end'] < adjusted[i - 1]['start'] + 0.3:
                adjusted[i - 1]['end'] = round(adjusted[i - 1]['start'] + 0.3, 3)

    # ── Fix short lines (same logic as generate_lyrics_timing post-pass) ──
    MIN_DUR = 1.4
    for i in range(len(adjusted)):
        dur = adjusted[i]['end'] - adjusted[i]['start']
        if dur >= MIN_DUR - 0.01:
            continue
        needed = MIN_DUR - dur
        earliest = adjusted[i - 1]['end'] + 0.05 if i > 0 else 0.0
        new_start = max(earliest, adjusted[i]['start'] - needed)
        adjusted[i]['start'] = round(new_start, 3)
        if adjusted[i]['end'] - adjusted[i]['start'] < MIN_DUR - 0.01:
            adjusted[i]['end'] = round(adjusted[i]['start'] + MIN_DUR + 0.01, 3)
        if i + 1 < len(adjusted) and adjusted[i]['end'] > adjusted[i + 1]['start'] - 0.05:
            adjusted[i]['end'] = round(adjusted[i + 1]['start'] - 0.05, 3)

    # ── Apply global offset ──
    if SUBTITLE_OFFSET != 0:
        for sub in adjusted:
            sub['start'] = round(max(0, sub['start'] + SUBTITLE_OFFSET), 3)
            sub['end']   = round(max(0.1, sub['end'] + SUBTITLE_OFFSET), 3)
        console.print(f"    Applied global offset: {SUBTITLE_OFFSET:+.2f}s")

    # Don't exceed total duration
    for sub in adjusted:
        sub['end'] = min(sub['end'], total_duration - 0.1)

    # Re-index
    for i, sub in enumerate(adjusted):
        sub['index'] = i + 1

    return adjusted


def apply_subtitle_offset(subtitles: list[dict], total_duration: float) -> list[dict]:
    """
    Apply a uniform time offset to all subtitles without onset snapping.

    Use this when timing comes from a hand-timed SRT file that already
    has correct *relative* timing — we only need to shift everything
    earlier to compensate for perceptual / rendering delay.
    """
    if SUBTITLE_OFFSET == 0:
        return subtitles

    adjusted = [sub.copy() for sub in subtitles]
    for sub in adjusted:
        sub['start'] = round(max(0, sub['start'] + SUBTITLE_OFFSET), 3)
        sub['end']   = round(max(0.1, sub['end'] + SUBTITLE_OFFSET), 3)

    # Fix any overlaps caused by the shift
    for i in range(1, len(adjusted)):
        if adjusted[i]['start'] < adjusted[i - 1]['end']:
            adjusted[i - 1]['end'] = round(adjusted[i]['start'] - 0.02, 3)
            if adjusted[i - 1]['end'] < adjusted[i - 1]['start'] + 0.3:
                adjusted[i - 1]['end'] = round(adjusted[i - 1]['start'] + 0.3, 3)

    # Don't exceed total duration
    for sub in adjusted:
        sub['end'] = min(sub['end'], total_duration - 0.1)

    # Re-index
    for i, sub in enumerate(adjusted):
        sub['index'] = i + 1

    console.print(f"    Applied uniform offset: {SUBTITLE_OFFSET:+.2f}s (no onset snapping)")
    return adjusted


def create_timed_subtitles(lyrics_lines: list[dict], total_duration: float) -> list[dict]:
    """
    Distribute lyrics across the audio duration.
    
    Strategy:
    - Leave ~5% intro silence and ~5% outro silence
    - Stage directions get shorter display time
    - Blank lines create pauses
    - Lyric lines get proportional time based on text length
    - Group short consecutive lines together for natural phrasing
    """
    # Filter to get displayable content
    content_lines = []
    for i, line in enumerate(lyrics_lines):
        if line['is_blank']:
            content_lines.append({'text': '', 'type': 'pause', 'weight': 0.3})
        elif line['is_direction']:
            content_lines.append({'text': line['text'], 'type': 'direction', 'weight': 0.5})
        else:
            # Weight by text length (longer lines need more reading time)
            word_count = len(line['text'].split())
            weight = max(1.0, word_count * 0.35)
            content_lines.append({'text': line['text'], 'type': 'lyric', 'weight': weight})

    if not content_lines:
        return []

    # Reserve time for intro/outro silence
    intro_time = min(total_duration * 0.06, 8.0)  # Up to 8s intro
    outro_time = min(total_duration * 0.06, 8.0)   # Up to 8s outro
    available_time = total_duration - intro_time - outro_time

    # Calculate total weight
    total_weight = sum(cl['weight'] for cl in content_lines)
    if total_weight == 0:
        return []

    time_per_weight = available_time / total_weight

    # Assign timestamps
    subtitles = []
    current_time = intro_time
    sub_index = 1

    for cl in content_lines:
        duration = cl['weight'] * time_per_weight
        
        if cl['type'] == 'pause':
            # Pauses just advance time, no subtitle shown
            current_time += duration
            continue
        
        if cl['type'] == 'direction':
            # Show directions briefly in italics
            start = current_time
            end = current_time + duration
            subtitles.append({
                'index': sub_index,
                'start': start,
                'end': end,
                'text': cl['text']
            })
            sub_index += 1
            current_time = end
            continue

        # Regular lyric line
        start = current_time
        end = current_time + duration
        
        # Ensure minimum display time of 1.5s and max of 6s
        actual_duration = end - start
        if actual_duration < 1.5:
            end = start + 1.5
        elif actual_duration > 6.0:
            end = start + 6.0

        # Don't exceed total duration
        if end > total_duration - 1.0:
            end = total_duration - 1.0
        if start >= end:
            continue

        subtitles.append({
            'index': sub_index,
            'start': start,
            'end': end,
            'text': cl['text']
        })
        sub_index += 1
        current_time = end

    return subtitles


def parse_srt_time(time_str: str) -> float:
    """Parse SRT timestamp 'HH:MM:SS,mmm' to seconds."""
    time_str = time_str.strip()
    h, m, rest = time_str.split(':')
    s, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_path: str) -> list[dict]:
    """
    Parse an existing SRT file into subtitle dicts.
    Returns list of {'index': int, 'start': float, 'end': float, 'text': str}.
    """
    subtitles = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split on double-newline to get blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        # Parse timestamps
        ts_match = re.match(r'(\S+)\s*-->\s*(\S+)', lines[1])
        if not ts_match:
            continue
        start = parse_srt_time(ts_match.group(1))
        end = parse_srt_time(ts_match.group(2))
        text = ' '.join(l.strip() for l in lines[2:]).strip()
        if text:
            subtitles.append({
                'index': index,
                'start': start,
                'end': end,
                'text': text,
            })
    return subtitles


def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(subtitles: list[dict], srt_path: str):
    """Write subtitles to an SRT file."""
    with open(srt_path, 'w', encoding='utf-8') as f:
        for sub in subtitles:
            f.write(f"{sub['index']}\n")
            f.write(f"{format_srt_time(sub['start'])} --> {format_srt_time(sub['end'])}\n")
            f.write(f"{sub['text']}\n\n")


def _shift_clips(clips, global_start):
    """Shift clip start times by global_start and sync masks."""
    shifted = []
    for clip in clips:
        c = clip.with_start(clip.start + global_start)
        if c.mask is not None:
            c.mask = c.mask.with_start(c.start)
        shifted.append(c)
    return shifted


def build_subtitle_clips(subtitles: list[dict], width: int, height: int) -> list:
    """
    Build FancyText subtitle clips with white text, drop shadow, stroke,
    and red word highlighting using Guttural.ttf font.
    Returns a list of moviepy clips ready for compositing.
    """
    # Adaptive highlight: the singer typically fills ~60% of the gap
    # between line starts with actual words; the remaining ~40% is
    # breath / transition silence.  This fraction adapts highlight pace
    # to the real tempo instead of using a fixed per-word rate.
    SINGING_FILL = 0.60
    MIN_PER_WORD = 0.15   # ceiling speed (~7 words/sec)
    MAX_PER_WORD = 0.50   # floor speed  (~2 words/sec)

    all_clips = []
    for sub in subtitles:
        sub_duration = sub['end'] - sub['start']
        if sub_duration <= 0:
            continue

        text = sub['text']
        if not text.strip():
            continue

        # Adaptive highlight duration derived from the actual inter-line
        # gap.  This naturally matches the singer's tempo: tight lines
        # get fast highlights, spacious lines get slower ones.
        n_words = len(text.split())
        singing_time = sub_duration * SINGING_FILL
        # Clamp per-word pace to sane bounds
        per_word = singing_time / max(n_words, 1)
        per_word = max(MIN_PER_WORD, min(per_word, MAX_PER_WORD))
        singing_time = n_words * per_word
        highlight_dur = max(0.8, min(singing_time, sub_duration * 0.95))

        try:
            word_clips = create_word_fancytext_adv(
                text=text,
                duration=highlight_dur,
                width=width,
                height=height,
                font_size=FONT_SIZE,
                font=FONT_PATH,
                text_color=(255, 255, 255),            # white text
                highlight_color=(220, 20, 20),         # red word highlight
                stroke_color=(0, 0, 0),                # black stroke
                stroke_width=3,
                max_words_per_line=6,
                position_y_ratio=SUBTITLE_Y_RATIO,
                shadow_enabled=True,
                shadow_offset=(3, 3),
                shadow_blur=5,
            )

            # Extend the last clip so the text (with last word highlighted)
            # stays visible for the full subtitle duration instead of
            # disappearing when the fast highlight pass finishes.
            if word_clips and sub_duration > highlight_dur:
                extra = sub_duration - highlight_dur
                last = word_clips[-1]
                new_dur = last.duration + extra
                word_clips[-1] = last.with_duration(new_dur)
                if word_clips[-1].mask is not None:
                    word_clips[-1].mask = word_clips[-1].mask.with_duration(new_dur)

            # FancyText has an internal 5% lead-in (start_offset = duration * 0.05)
            # before the first word highlights.  Shift clips back so the first
            # highlighted word appears exactly at sub['start'].
            fancytext_lead_in = highlight_dur * 0.05
            all_clips.extend(_shift_clips(word_clips, sub['start'] - fancytext_lead_in))
        except Exception as e:
            console.print(f"    [yellow]Warning: FancyText failed for '{text[:30]}...': {e}[/yellow]")

    return all_clips


def create_video(song_name: str, txt_path: str, wav_path: str, bg_path: str, output_path: str):
    """Create a video for a single song with styled subtitles."""
    console.print(f"\n[bold cyan]Processing:[/bold cyan] {song_name}")

    # Get audio duration
    duration = get_wav_duration(wav_path)
    console.print(f"  Audio duration: {int(duration // 60)}m {duration % 60:.1f}s")

    # Always generate timing from lyrics + audio analysis
    lyrics = parse_lyrics(txt_path)
    srt_output = str(OUTPUT_DIR / f"{song_name}.srt")
    subtitles = generate_lyrics_timing(lyrics, wav_path, duration)

    # Snap subtitle start times to actual phrase onsets in the audio
    subtitles = align_subtitles_to_audio(subtitles, wav_path, duration)

    # Write generated SRT alongside the video output
    write_srt(subtitles, srt_output)
    console.print(f"  SRT saved: {srt_output}")

    # Load background, fit to frame preserving aspect ratio, pad with black
    bg_image = Image.open(bg_path).convert('RGB')
    src_w, src_h = bg_image.size
    scale = min(VIDEO_WIDTH / src_w, VIDEO_HEIGHT / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    bg_image = bg_image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0))
    canvas.paste(bg_image, ((VIDEO_WIDTH - new_w) // 2, (VIDEO_HEIGHT - new_h) // 2))
    bg_array = np.array(canvas)

    # Create background clip
    bg_clip = ImageClip(bg_array).with_duration(duration)

    # Build subtitle clips
    console.print(f"  Building FancyText subtitle clips...")
    subtitle_clips = build_subtitle_clips(subtitles, VIDEO_WIDTH, VIDEO_HEIGHT)
    console.print(f"  Created {len(subtitle_clips)} FancyText word clips")

    # Load audio
    audio_clip = AudioFileClip(wav_path)

    # Composite everything
    final = CompositeVideoClip(
        [bg_clip, *subtitle_clips],
        size=(VIDEO_WIDTH, VIDEO_HEIGHT),
    ).with_duration(duration).with_fps(VIDEO_FPS)
    final = final.with_audio(audio_clip)

    # Write output
    console.print(f"  [yellow]Rendering video...[/yellow]")
    final.write_videofile(
        output_path,
        fps=VIDEO_FPS,
        codec='libx264',
        audio_codec='aac',
        audio_bitrate='192k',
        preset='medium',
        threads=4,
        logger='bar'
    )
    console.print(f"  [bold green]Done:[/bold green] {output_path}")

    # Cleanup
    audio_clip.close()


def main():
    """Process all songs in the Album folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bg_path = str(ALBUM_DIR / "Background.png")
    if not os.path.exists(bg_path):
        console.print("[bold red]Error:[/bold red] Background.png not found in Album folder!")
        return

    # Find all song pairs (txt + wav)
    txt_files = sorted(ALBUM_DIR.glob("*.txt"))
    songs = []
    for txt_file in txt_files:
        wav_file = txt_file.with_suffix('.wav')
        if wav_file.exists():
            song_name = txt_file.stem
            songs.append((song_name, str(txt_file), str(wav_file)))
        else:
            console.print(f"[yellow]Warning:[/yellow] No WAV found for {txt_file.name}, skipping")

    console.print(f"\n[bold]Found {len(songs)} songs to process:[/bold]")
    for i, (name, _, _) in enumerate(songs, 1):
        console.print(f"  {i}. {name}")

    # Process each song
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Creating album videos...", total=len(songs))

        for song_name, txt_path, wav_path in songs:
            output_path = str(OUTPUT_DIR / f"{song_name}.mp4")
            try:
                create_video(song_name, txt_path, wav_path, bg_path, output_path)
                successful += 1
            except Exception as e:
                console.print(f"  [bold red]Error processing {song_name}:[/bold red] {e}")
                import traceback
                traceback.print_exc()
                failed += 1
            progress.advance(task)

    console.print(f"\n[bold]{'='*50}[/bold]")
    console.print(f"[bold green]Completed:[/bold green] {successful} videos created")
    if failed:
        console.print(f"[bold red]Failed:[/bold red] {failed} videos")
    console.print(f"[bold]Output directory:[/bold] {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
