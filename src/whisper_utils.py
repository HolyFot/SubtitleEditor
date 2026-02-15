"""
Whisper-based voice recognition and lyrics alignment utilities.
"""
import re
from difflib import SequenceMatcher

from rich.console import Console

# Whisper for voice recognition
try:
    from faster_whisper import WhisperModel
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False
    WhisperModel = None

console = Console()

# Pre-roll: show subtitle this many seconds BEFORE the detected vocal onset.
SUBTITLE_PRE_ROLL = 0.20

# Global model cache to avoid reloading
_whisper_model = None


def _get_whisper_model(model_size: str = "base"):
    """Get or create a cached Whisper model instance."""
    global _whisper_model
    if _whisper_model is None:
        if not _HAS_WHISPER:
            raise RuntimeError("faster-whisper not installed")
        console.print(f"[cyan]Loading Whisper model ({model_size})...[/cyan]")
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_with_whisper(
    wav_path: str,
    language: str | None = None,
    model_size: str = "base"
) -> list[dict]:
    """
    Transcribe audio using Whisper and return word-level timestamps.
    
    Parameters
    ----------
    wav_path : str
        Path to WAV file
    language : str, optional
        Language code (e.g., 'en', 'es'). Auto-detected if None.
    model_size : str
        Whisper model size: 'tiny', 'base', 'small', 'medium', 'large-v3'
    
    Returns
    -------
    list[dict]
        List of words with timestamps:
        [{'word': str, 'start': float, 'end': float, 'confidence': float}, ...]
    """
    if not _HAS_WHISPER:
        console.print("[yellow]Warning: faster-whisper not available[/yellow]")
        return []
    
    model = _get_whisper_model(model_size)
    
    console.print(f"[cyan]Transcribing audio with Whisper...[/cyan]")
    
    # First try: Transcribe WITHOUT VAD filter (better for music with instruments)
    # VAD can filter out vocals that are mixed with guitars/drums
    segments, info = model.transcribe(
        wav_path,
        language=language,
        word_timestamps=True,
        vad_filter=False,  # Disabled - VAD often misses vocals in music
    )
    
    # Extract word-level timestamps
    words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                words.append({
                    'word': word.word.strip(),
                    'start': word.start,
                    'end': word.end,
                    'confidence': word.probability,
                })
    
    # If we got very few words, the audio might need VAD to filter noise
    # Try again with VAD in that case
    if len(words) < 20:
        console.print(f"[yellow]Only {len(words)} words detected, retrying with VAD...[/yellow]")
        segments, info = model.transcribe(
            wav_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=1000,
                speech_pad_ms=400,
                threshold=0.3,
                min_speech_duration_ms=100,
            ),
        )
        words = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    words.append({
                        'word': word.word.strip(),
                        'start': word.start,
                        'end': word.end,
                        'confidence': word.probability,
                    })
    
    if language is None:
        console.print(f"[dim]Detected language: {info.language} "
                      f"(probability: {info.language_probability:.2f})[/dim]")
    
    if words:
        first_word = words[0]
        last_word = words[-1]
        console.print(f"[green]Transcribed {len(words)} words "
                      f"({first_word['start']:.1f}s - {last_word['end']:.1f}s)[/green]")
    else:
        console.print(f"[yellow]No words transcribed[/yellow]")
    return words


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', '', text.lower())
    return ' '.join(text.split())


def _word_similarity(word1: str, word2: str) -> float:
    """Calculate similarity between two words (0-1)."""
    w1 = _normalize_text(word1)
    w2 = _normalize_text(word2)
    if not w1 or not w2:
        return 0.0
    return SequenceMatcher(None, w1, w2).ratio()


def align_lyrics_to_whisper(
    subtitles: list[dict],
    whisper_words: list[dict],
    total_duration: float,
    vocal_start: float = 0.0
) -> list[dict]:
    """
    Align lyrics to Whisper transcription using fuzzy matching.
    
    Uses dynamic programming to find the optimal alignment between
    the provided lyrics and the transcribed words, handling:
    - Minor transcription errors
    - Words split differently (e.g., "gonna" vs "going to")
    - Missing/extra words in transcription
    
    Parameters
    ----------
    subtitles : list[dict]
        Original subtitles with 'text', 'start', 'end' keys
    whisper_words : list[dict]
        Whisper transcription from transcribe_with_whisper()
    total_duration : float
        Total audio duration in seconds
    vocal_start : float
        User-specified time when vocals actually start (used to anchor alignment)
    
    Returns
    -------
    list[dict]
        Subtitles with updated start/end times based on Whisper alignment
    """
    if not whisper_words:
        console.print("[yellow]No Whisper words to align to[/yellow]")
        return subtitles
    
    # Show Whisper coverage info
    whisper_end_time = whisper_words[-1]['end'] if whisper_words else 0
    console.print(f"[cyan]Aligning {len(subtitles)} lyrics to {len(whisper_words)} "
                  f"Whisper words (up to {whisper_end_time:.1f}s)[/cyan]")
    
    adjusted = [sub.copy() for sub in subtitles]
    
    # Identify direction markers (non-singing text)
    def _is_direction(text: str) -> bool:
        t = text.strip()
        return ((t.startswith('[') and t.endswith(']'))
                or (t.startswith('(') and t.endswith(')'))
                or t.startswith('(') and t.endswith('):'))
    
    # Track timing offset from energy-based alignment to Whisper
    # This helps when energy-based is way off (e.g., placed lyrics during intro)
    # If user provided vocal_start, calculate offset from first Whisper word
    timing_offset = 0.0
    offset_established = False
    whisper_too_late = False  # Flag: Whisper missed early vocals
    
    # If user specified vocal_start, establish offset immediately
    if vocal_start > 0:
        # Find first actual lyric (not a direction marker)
        first_lyric_orig_time = 0
        for sub in adjusted:
            if not _is_direction(sub['text']):
                first_lyric_orig_time = sub['start']
                break
        
        # Calculate offset so first lyric lands at vocal_start
        timing_offset = vocal_start - first_lyric_orig_time
        offset_established = True
        
        first_whisper_time = whisper_words[0]['start'] if whisper_words else 0
        console.print(f"[cyan]Vocal start: {vocal_start:.1f}s, "
                      f"first lyric orig: {first_lyric_orig_time:.1f}s, "
                      f"Whisper starts: {first_whisper_time:.1f}s, "
                      f"applying offset: {timing_offset:+.1f}s[/cyan]")
        
        # If Whisper's first word is much later than vocal_start,
        # Whisper missed the early vocals - don't trust Whisper timing
        if first_whisper_time > vocal_start + 5.0:
            whisper_too_late = True
            console.print(f"[yellow]Whisper starts {first_whisper_time - vocal_start:.1f}s "
                          f"after vocal_start - using offset-based timing[/yellow]")
    
    # Track last matched whisper word to enforce sequential constraint
    last_matched_whisper_idx = 0
    
    # Track actual aligned timing for gap validation
    last_aligned_end = 0.0
    last_orig_end = 0.0
    
    # Process each subtitle line
    for sub_idx, sub in enumerate(adjusted):
        if _is_direction(sub['text']):
            continue
        sub_words = _normalize_text(sub['text']).split()
        if not sub_words:
            continue
        if sub_idx >= 30:
            console.print(f"[dim]Processing line {sub_idx}: '{sub['text'][:40]}...'[/dim]")
        orig_start = sub['start']
        orig_end = sub['end']

        # Always attempt Whisper matching, even if Whisper starts late
        best_match_start = None
        best_match_end = None
        best_score = 0.0

        # Dynamic search window: expand for early lines, shrink for later
        if offset_established and last_aligned_end > 0:
            expected_gap = orig_start - last_orig_end
            expected_start = last_aligned_end + expected_gap
            search_center = expected_start
            search_margin = 30.0 if sub_idx < 5 else 20.0
        elif offset_established:
            search_center = orig_start + timing_offset
            search_margin = 40.0 if sub_idx < 5 else 25.0
        else:
            search_center = orig_start
            search_margin = total_duration

        candidate_indices = [
            i for i, w in enumerate(whisper_words)
            if i >= last_matched_whisper_idx
            and w['start'] >= search_center - search_margin
            and w['start'] <= search_center + search_margin
        ]
        if not candidate_indices and last_matched_whisper_idx < len(whisper_words):
            candidate_indices = [
                i for i, w in enumerate(whisper_words)
                if i >= last_matched_whisper_idx
                and w['start'] <= search_center + 60.0
            ]

        for start_idx in candidate_indices:
            max_whisper_words = min(len(sub_words) * 2, len(whisper_words) - start_idx)
            for num_words in range(1, max_whisper_words + 1):
                end_idx = start_idx + num_words
                if end_idx > len(whisper_words):
                    break
                whisper_text = ' '.join(w['word'] for w in whisper_words[start_idx:end_idx])
                sub_text = ' '.join(sub_words)
                score = _word_similarity(sub_text, whisper_text)
                # Boost for exact match
                if whisper_text == sub_text:
                    score += 1.0
                if num_words == len(sub_words):
                    score *= 1.2
                whisper_start = whisper_words[start_idx]['start']
                time_distance = abs(whisper_start - search_center)
                # Penalize timing mismatches more strongly for early lines
                if time_distance < 8.0:
                    score += 0.7 * (1.0 - time_distance / 8.0)
                elif time_distance > 8.0:
                    score -= 0.3 * ((time_distance - 8.0) / 20.0)
                if score > best_score:
                    best_score = score
                    best_match_start = start_idx
                    best_match_end = end_idx

        min_score = 1.3
        global_min_score = 0.85  # Lower threshold for global search (fallback)
        used_whisper = False
        if best_match_start is not None and best_score >= min_score:
            new_start = whisper_words[best_match_start]['start']
            new_end = whisper_words[best_match_end - 1]['end']
            last_matched_whisper_idx = best_match_end
            if not offset_established:
                timing_offset = new_start - sub['start']
                offset_established = True
                console.print(f"[cyan]Timing offset established: {timing_offset:+.1f}s[/cyan]")
            last_aligned_end = new_end
            last_orig_end = orig_end
            adjusted[sub_idx]['start'] = round(max(0, new_start - SUBTITLE_PRE_ROLL), 3)
            adjusted[sub_idx]['end'] = round(min(total_duration, new_end + 0.1), 3)
            used_whisper = True
            console.print(f"[green]Aligned: '{sub['text'][:30]}...' -> "
                          f"{adjusted[sub_idx]['start']:.2f}s (score: {best_score:.2f})[/green]")
        else:
            # Global search for a match anywhere in the audio
            # Use lower threshold since we're already in fallback territory -
            # any reasonable Whisper match is better than offset-based guessing
            global_best_score = 0.0
            global_best_start = None
            global_best_end = None
            for start_idx in range(len(whisper_words)):
                max_whisper_words = min(len(sub_words) * 2, len(whisper_words) - start_idx)
                for num_words in range(1, max_whisper_words + 1):
                    end_idx = start_idx + num_words
                    if end_idx > len(whisper_words):
                        break
                    whisper_text = ' '.join(w['word'] for w in whisper_words[start_idx:end_idx])
                    sub_text = ' '.join(sub_words)
                    score = _word_similarity(sub_text, whisper_text)
                    if whisper_text == sub_text:
                        score += 1.0
                    if num_words == len(sub_words):
                        score *= 1.2
                    # No timing bonus/penalty for global search
                    if score > global_best_score:
                        global_best_score = score
                        global_best_start = start_idx
                        global_best_end = end_idx
            if global_best_start is not None and global_best_score >= global_min_score:
                new_start = whisper_words[global_best_start]['start']
                new_end = whisper_words[global_best_end - 1]['end']
                last_matched_whisper_idx = global_best_end
                last_aligned_end = new_end
                last_orig_end = orig_end
                adjusted[sub_idx]['start'] = round(max(0, new_start - SUBTITLE_PRE_ROLL), 3)
                adjusted[sub_idx]['end'] = round(min(total_duration, new_end + 0.1), 3)
                console.print(f"[green]Global Aligned: '{sub['text'][:30]}...' -> "
                              f"{adjusted[sub_idx]['start']:.2f}s (score: {global_best_score:.2f})[/green]")
            else:
                # Fallback: offset-based timing, but log score (show the actual fallback score)
                if last_aligned_end > 0:
                    expected_gap = orig_start - last_orig_end
                    # If the original gap is large (>5s), preserve it in fallback
                    min_gap = 5.0
                    if expected_gap > min_gap:
                        fallback_start = last_aligned_end + expected_gap
                    else:
                        fallback_start = last_aligned_end + max(expected_gap, 0.0)
                    fallback_end = fallback_start + (orig_end - orig_start)
                else:
                    fallback_start = orig_start + timing_offset
                    fallback_end = orig_end + timing_offset
                adjusted[sub_idx]['start'] = round(max(0, fallback_start - SUBTITLE_PRE_ROLL), 3)
                adjusted[sub_idx]['end'] = round(min(total_duration, fallback_end + 0.1), 3)
                last_aligned_end = fallback_end
                last_orig_end = orig_end
                fallback_log_score = max(best_score, global_best_score)
                console.print(f"[yellow]Fallback: '{sub['text'][:30]}...' -> "
                              f"{adjusted[sub_idx]['start']:.2f}s (score: {fallback_log_score:.2f})[/yellow]")

    # Apply timing offset to direction markers (bracketed text like [Intro], [Verse])
    # These were skipped during alignment but should still be shifted
    if offset_established and timing_offset != 0:
        for sub in adjusted:
            if _is_direction(sub['text']):
                sub['start'] = round(max(0, sub['start'] + timing_offset), 3)
                sub['end'] = round(min(total_duration, sub['end'] + timing_offset), 3)
    
    # Fix overlaps
    for i in range(1, len(adjusted)):
        if adjusted[i]['start'] < adjusted[i - 1]['end']:
            # Split the difference
            mid = (adjusted[i]['start'] + adjusted[i - 1]['end']) / 2
            adjusted[i - 1]['end'] = round(mid - 0.02, 3)
            adjusted[i]['start'] = round(mid + 0.02, 3)
    
    # Ensure minimum duration
    MIN_DUR = 1.4
    for sub in adjusted:
        if sub['end'] - sub['start'] < MIN_DUR:
            sub['end'] = round(min(total_duration, sub['start'] + MIN_DUR), 3)
    
    # Fix any new overlaps from duration extension
    for i in range(len(adjusted) - 1):
        if adjusted[i]['end'] > adjusted[i + 1]['start']:
            adjusted[i]['end'] = round(adjusted[i + 1]['start'] - 0.02, 3)
    
    # Re-index
    for i, sub in enumerate(adjusted):
        sub['index'] = i + 1
    
    return adjusted


def has_whisper() -> bool:
    """Check if faster-whisper is available."""
    return _HAS_WHISPER
