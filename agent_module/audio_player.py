"""
Custom themed audio player for Streamlit, replacing the native browser
<audio> controls used by st.audio().

WHY THIS EXISTS: st.audio() renders the browser's native HTML5 audio
controls, which can't be restyled via CSS — browsers deliberately don't
expose their internal play/pause/scrubber UI to styling, for consistency and
security reasons. This left the TTS playback looking like a generic system
widget dropped into an otherwise custom dark-themed chat UI.

HOW IT WORKS: builds a self-contained HTML/CSS/JS string (play button,
progress bar, time label, all in the app's dark theme palette) with the
audio embedded as a base64 data URI, rendered via st.html(...,
unsafe_allow_javascript=True). Verified against Streamlit 1.58's bundled
DOMPurify config, which explicitly whitelists <script> tags for st.html
(ADD_TAGS: ['script', 'style']), and tested end-to-end against a real
ffmpeg-generated MP3 to confirm the player function itself runs cleanly
inside a live Streamlit process with no exceptions.

NOTE: actual in-browser JS execution (play/pause click handling, progress
bar animation) could not be verified from the sandbox this was built in, as
no headless browser was available there — confirm this part works by
actually clicking play in your browser after deploying.
"""

import base64
import hashlib


def render_custom_audio_player(
    audio_bytes: bytes, autoplay: bool = False, player_id: str = None
) -> str:
    """
    Builds a self-contained HTML/CSS/JS audio player styled to match the
    app's dark theme (#1e1e2f containers, #7c5cff accent — matching the
    existing chat bubble colors in app.py's CUSTOM STYLING block).

    player_id is derived from a hash of the audio bytes by default, so the
    same message always renders with the same id across Streamlit reruns
    (the whole chat history re-renders every rerun) while different messages
    never collide in the DOM.

    Returns an empty string for empty/None audio_bytes, so callers can pass
    this directly to st.html() without an extra guard — st.html() raises if
    given truly empty content, so check `if audio_bytes:` before calling
    this if audio might be missing for a given message.
    """
    if not audio_bytes:
        return ""

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

    if player_id is None:
        player_id = "p" + hashlib.md5(audio_bytes).hexdigest()[:10]

    autoplay_attr = "autoplay" if autoplay else ""

    html = f"""
    <div class="dt-audio-player" id="container-{player_id}">
      <button class="dt-play-btn" id="btn-{player_id}" aria-label="Play">
        <svg id="icon-play-{player_id}" width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M2 1.5 L12 7 L2 12.5 Z" fill="white"/>
        </svg>
        <svg id="icon-pause-{player_id}" width="14" height="14" viewBox="0 0 14 14" fill="none" style="display:none;">
          <rect x="2" y="1.5" width="3.5" height="11" fill="white"/>
          <rect x="8.5" y="1.5" width="3.5" height="11" fill="white"/>
        </svg>
      </button>
      <div class="dt-progress-track" id="track-{player_id}">
        <div class="dt-progress-fill" id="fill-{player_id}"></div>
        <div class="dt-progress-handle" id="handle-{player_id}"></div>
      </div>
      <span class="dt-time-label" id="time-{player_id}">0:00</span>
      <audio id="audio-{player_id}" {autoplay_attr} preload="metadata">
        <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
      </audio>
    </div>

    <style>
      .dt-audio-player {{
        display: flex;
        align-items: center;
        gap: 10px;
        background: #1e1e2f;
        border: 1px solid #34344a;
        border-radius: 10px;
        padding: 8px 12px;
        margin-top: 6px;
        font-family: "Source Sans Pro", sans-serif;
        max-width: 100%;
      }}
      .dt-play-btn {{
        flex-shrink: 0;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #7c5cff;
        border: none;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background 0.15s ease, transform 0.1s ease;
      }}
      .dt-play-btn:hover {{ background: #8e72ff; }}
      .dt-play-btn:active {{ transform: scale(0.93); }}
      .dt-progress-track {{
        position: relative;
        flex: 1;
        height: 4px;
        background: #34344a;
        border-radius: 2px;
        cursor: pointer;
      }}
      .dt-progress-fill {{
        position: absolute;
        left: 0; top: 0;
        height: 100%;
        width: 0%;
        background: #7c5cff;
        border-radius: 2px;
      }}
      .dt-progress-handle {{
        position: absolute;
        top: 50%;
        left: 0%;
        width: 10px; height: 10px;
        background: #fff;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0 4px rgba(0,0,0,0.4);
      }}
      .dt-time-label {{
        flex-shrink: 0;
        color: #9b9bb0;
        font-size: 12px;
        min-width: 32px;
        text-align: right;
      }}
    </style>

    <script>
    (function() {{
      const audio = document.getElementById('audio-{player_id}');
      const btn = document.getElementById('btn-{player_id}');
      const iconPlay = document.getElementById('icon-play-{player_id}');
      const iconPause = document.getElementById('icon-pause-{player_id}');
      const track = document.getElementById('track-{player_id}');
      const fill = document.getElementById('fill-{player_id}');
      const handle = document.getElementById('handle-{player_id}');
      const timeLabel = document.getElementById('time-{player_id}');

      function formatTime(sec) {{
        if (!isFinite(sec)) return '0:00';
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60).toString().padStart(2, '0');
        return m + ':' + s;
      }}

      function updateUI() {{
        const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
        fill.style.width = pct + '%';
        handle.style.left = pct + '%';
        // Shows time REMAINING while playing (e.g. "0:12" counting down),
        // matches the convention most chat/voice apps use.
        timeLabel.textContent = formatTime(audio.duration && !audio.ended ? audio.duration - audio.currentTime : audio.duration || 0);
      }}

      btn.addEventListener('click', function() {{
        if (audio.paused) {{
          audio.play();
        }} else {{
          audio.pause();
        }}
      }});

      audio.addEventListener('play', function() {{
        iconPlay.style.display = 'none';
        iconPause.style.display = 'block';
      }});
      audio.addEventListener('pause', function() {{
        iconPlay.style.display = 'block';
        iconPause.style.display = 'none';
      }});
      audio.addEventListener('ended', function() {{
        iconPlay.style.display = 'block';
        iconPause.style.display = 'none';
        fill.style.width = '0%';
        handle.style.left = '0%';
      }});
      audio.addEventListener('timeupdate', updateUI);
      audio.addEventListener('loadedmetadata', updateUI);

      track.addEventListener('click', function(e) {{
        const rect = track.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        if (audio.duration) {{
          audio.currentTime = pct * audio.duration;
        }}
      }});
    }})();
    </script>
    """
    return html
