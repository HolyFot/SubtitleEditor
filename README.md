# SubEditor — Subtitle Video Editor

A desktop application for generating/editing videos with subtitles auto synced to lyrics. Built with Python, has different subtitle presets (editable), multi-track timeline editing, cool audio effects and video export.

## Features

### Timeline Editor
- **Multi-track timeline** with video, audio, and subtitle tracks
- **Drag-and-drop** clip editing with resize handles
- **Real-time waveform** visualization for audio tracks
- **Video thumbnail** previews on the video track
- **Zoom and scroll** for precise editing
- **Playhead scrubbing** with frame-accurate preview

### Subtitle Styling
- **Karaoke-style highlighting** (word or character mode)
- **Rich text effects**: stroke, shadow, glow, gradient, bevel, chrome
- **Texture overlays** with blend modes
- **Custom fonts** with automatic system font detection
- **Preset system** for saving and loading styles
- **Position and justification** controls

### Audio Processing
- **10-band parametric EQ** with presets (Bass Boost, Vocal Enhance, etc.)
- **Audio effects**: pitch shift, reverb, chorus, compression, autotune, vocoders, delay
- **Multi-layer audio** support

### Export
- **Multiple formats**: MP4, AVI, MKV, WebM, MOV, GIF
- **Configurable codecs**: H.264, H.265, VP9, ProRes, etc.
- **Quality presets** with CRF control
- **Audio codec options**: AAC, MP3, Opus, FLAC
- **Progress tracking** with elapsed time and ETA

### Project Management
- **Save/load projects** with all settings preserved
- **SRT import/export** for subtitle interchange
- **Auto-alignment** of lyrics to audio using speech recognition

Note: you can use \r in a subtitle to make a new line.

## Installation

1. Clone the repository (or just download):
```bash
git clone https://github.com/yourusername/SubtitleEditor.git
cd SubtitleEditor
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python src/subtitle_editor.py
```

## Usage

1. **Load Media**: Open an audio file (WAV) and optionally a video or background image
2. **Add Lyrics**: Load a lyrics text file or import an SRT
3. **Generate Subtitles**: Auto-generate timed subtitles from lyrics
4. **Style**: Customize appearance using the Style panel
5. **Edit Timeline**: Fine-tune timing by dragging clips on the timeline
6. **Preview**: Use the preview panel to see real-time results
7. **Export**: Render the final video with all effects applied

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `S` | Split clip at playhead |
| `Delete` | Delete selected clip |
| `Ctrl+N` | New project |
| `Ctrl+O` | Open project |
| `Ctrl+S` | Save project |
| `Ctrl+C` | Copy clip |
| `Ctrl+V` | Paste clip |
| `Ctrl+D` | Duplicate clip |