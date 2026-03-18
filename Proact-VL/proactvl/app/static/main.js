// /static/main.js
(() => {
  // ================= Debugging (set window.DEBUG=true in index.html) =================
  const DEBUG = !!window.DEBUG;
  const dlog  = (...a) => { if (DEBUG) console.log(...a); };
  const dwarn = (...a) => { if (DEBUG) console.warn(...a); };
  const derr  = (...a) => { if (DEBUG) console.error(...a); };

  dlog('[front] main.js loaded. WS_BASE =', window.WS_BASE || '(auto)');

  // ================= DOM =================
  const video = document.getElementById('preview');
  const canvas = document.getElementById('canvas'); // Hidden canvas: used only for frame capture
  const statusEl = document.getElementById('status');
  const commentaryBox = document.getElementById('commentary-box');

  const startWebcamBtn = document.getElementById('startWebcamBtn');
  const startScreenBtn = document.getElementById('startScreenBtn');
  const selectFileBtn  = document.getElementById('selectFileBtn');
  const fileInput      = document.getElementById('videoFileInput');

  const clearCommentaryBtn = document.getElementById('clearCommentaryBtn');
  const toggleCommentBtn = document.getElementById('toggleCommentBtn');
  const stopBtn          = document.getElementById('stopBtn');
  // userCommentForm
  const queryForm  = document.getElementById('queryForm');
  // userCommentInput
  const queryInput = document.getElementById('queryInput');

  // Model settings section
  const settingsStatus   = document.getElementById('settingsStatus');

  // ========== Assistant settings DOM ==========
  const assistantCountSelect = document.getElementById('assistantCountSelect');
  const applyCountBtn = document.getElementById('applyCountBtn');
  const assistantsConfigWrapper = document.getElementById('assistantsConfigWrapper');
  
  const assistantBlocks = [
    document.getElementById('assistant-block-1'),
    document.getElementById('assistant-block-2'),
    document.getElementById('assistant-block-3')
  ];
  const assistantFieldsets = [
    document.getElementById('settingsFieldset-1'),
    document.getElementById('settingsFieldset-2'),
    document.getElementById('settingsFieldset-3')
  ];

  // ================= Configuration =================
  const CAPTURE_INTERVAL_MS = 500;   // Frame capture: 2 FPS
  const INFER_INTERVAL_MS   = 1000;  // Inference: 1 Hz
  const JPEG_QUALITY        = 0.6;

  // WebSocket URL
  const BACKEND_WS_URL =
    window.WS_BASE ||
    ((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/stream');

  // ================= State =================
  let mediaStream = null;        // Webcam/screen stream
  let objectUrl   = null;        // Local video URL
  let sourceKind  = null;        // 'webcam' | 'screen' | 'file'

  let captureTimer = null;       // 2 FPS frame capture timer
  let inferTimer   = null;       // 1 Hz inference request timer
  let captureActive = false;     // Commentary toggle

  // Buffer for the last two frames
  let lastFrames = [];

  // WebSocket
  let ws = null;
  let reqSeq = 0;                // Auto-incrementing request ID
  const inflight = new Map();    // request_id -> { row, bubble }  (inference placeholder)
  const inflightCtl = new Map(); // request_id -> { okMsg }        (control message)

  // Optional query
  let pendingQuery = null;

  const JPEG_MIME = 'image/jpeg';

  // ================= Utility functions =================
  const clamp01 = (v) => {
    const n = Number(v);
    if (Number.isNaN(n)) return 0;
    return Math.min(1, Math.max(0, n));
  };

  const setStatus = (msg, ok) => {
    if (!statusEl) return;
    statusEl.textContent = msg || '';
    statusEl.classList.remove('status-ok', 'status-bad');
    if (ok === true) statusEl.classList.add('status-ok');
    else if (ok === false) statusEl.classList.add('status-bad');
  };

  const nowCNTime = () => {
    const d = new Date();
    return d.toLocaleTimeString('en-US', { hour12: false });
  };

  const setSettingsHint = (msg, ok = true) => {
    if (!settingsStatus) return;
    settingsStatus.textContent = msg || '';
    settingsStatus.style.color = ok ? '#16a34a' : '#dc2626';
  };

  const enableSources = (enabled) => {
    if (startWebcamBtn) startWebcamBtn.disabled = !enabled;
    if (startScreenBtn) startScreenBtn.disabled = !enabled;
    if (selectFileBtn)  selectFileBtn.disabled  = !enabled;
  };

  const enableReadyControls = (ready) => {
    if (toggleCommentBtn) toggleCommentBtn.disabled = !ready;
    if (stopBtn)          stopBtn.disabled          = !ready;
  };

  // Smart control for multiple Assistant fieldsets
  const setSettingsEnabled = (enabled) => {
    assistantFieldsets.forEach((fieldset, index) => {
      const block = assistantBlocks[index];
      if (!fieldset || !block) return;

      // If the block is visible, follow the global state; if hidden, force disable it
      if (block.style.display !== 'none') {
        fieldset.disabled = !enabled;
      } else {
        fieldset.disabled = true;
      }
    });

    if (!enabled) setSettingsHint('Pause commentary to adjust settings', false);
    else setSettingsHint('', true);
  };

  const setToggleUI = () => {
    if (!toggleCommentBtn) return;
    if (!captureActive) {
      toggleCommentBtn.textContent = 'Start Commentary';
      toggleCommentBtn.classList.remove('danger');
    } else {
      toggleCommentBtn.textContent = 'Pause Commentary';
      toggleCommentBtn.classList.add('danger');
    }
  };

  const cleanupStream = () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
  };

  const cleanupFile = () => {
    if (objectUrl) {
      try { URL.revokeObjectURL(objectUrl); } catch {}
      objectUrl = null;
    }
  };

  const resetVideo = () => {
    if (!video) return;
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();
  };

  function stopCaptureOnly(){
    captureActive = false;
    if (captureTimer) { clearInterval(captureTimer); captureTimer = null; }
    if (inferTimer)   { clearInterval(inferTimer);   inferTimer   = null; }
    setToggleUI();
  }

  function stopAll(message = 'Disconnected'){
    stopCaptureOnly();
    lastFrames = [];
    pendingQuery = null;
    cleanupStream();
    cleanupFile();
    resetVideo();
    enableSources(true);
    enableReadyControls(false);
    sourceKind = null;
    setStatus(message);
    setSettingsEnabled(true); 
    clearAudioQueue(true);
  }

  function tryAutoPlay(audio) {
    if (!audio) return;
    const playPromise = audio.play();
    if (playPromise && typeof playPromise.then === 'function') {
      playPromise
        .then(() => { dlog('[audio] autoplay ok'); })
        .catch((err) => { dwarn('[audio] autoplay blocked', err); });
    }
  }

  // ========== Global audio playback ==========
  const audioQueue = [];
  let globalAudio = null;
  let audioPlaying = false;

  function getGlobalAudio() {
    if (!globalAudio) {
      globalAudio = new Audio();
      globalAudio.preload = 'none';
      const onEndedOrError = () => { audioPlaying = false; playNextAudio(); };
      globalAudio.addEventListener('ended', onEndedOrError);
      globalAudio.addEventListener('error', onEndedOrError);
    }
    return globalAudio;
  }

  function enqueueAudio(audioUrl) {
    if (!audioUrl) return;
    audioQueue.push(audioUrl);
    if (!audioPlaying) playNextAudio();
  }

  function playNextAudio() {
    if (audioPlaying) return;
    if (!audioQueue.length) return;
    const url = audioQueue.shift();
    const audio = getGlobalAudio();
    audio.src = url;
    audioPlaying = true;
    tryAutoPlay(audio);
  }

  function clearAudioQueue(stopCurrent = true) {
    audioQueue.length = 0;
    if (stopCurrent && globalAudio) {
      try { globalAudio.pause(); globalAudio.currentTime = 0; } catch (e) {}
      audioPlaying = false;
    }
  }

  function clearCommentaryBox() {
    if (commentaryBox) commentaryBox.innerHTML = '';
  }

  // ========== Message rendering ==========
  function appendMessage(role, text, opts = {}) {
    const { speaker, audioUrl } = opts;
    const row = document.createElement('div');
    row.className = `msg msg-${role}`;

    // Fix: use a strict check so 0 is not treated as false
    if (speaker !== undefined && speaker !== null) {
      // Handle style compatibility like speaker_0 -> speaker-speaker-0
      const cls = 'speaker-' + String(speaker).toLowerCase().replace(/[^a-z0-9]/g, '-');
      row.classList.add(cls);
    }

    const meta = document.createElement('div');
    meta.className = 'meta';
    // const who = role === 'user' ? 'User' : (speaker || 'Model');
    const who = role === 'user' ? 'User' : (speaker !== undefined && speaker !== null ? speaker : 'Model');
    meta.textContent = `${who} · ${nowCNTime()}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;

    if (audioUrl) {
      const br = document.createElement('br');
      const tag = document.createElement('span');
      tag.className = 'audio-tag';
      tag.textContent = '🔊 Audio added to playback queue';
      tag.style.cursor = 'pointer';
      tag.title = 'Click to replay this sentence';
      tag.addEventListener('click', () => enqueueAudio(audioUrl));
      bubble.appendChild(br);
      bubble.appendChild(tag);
      enqueueAudio(audioUrl);
    }

    row.appendChild(meta);
    row.appendChild(bubble);
    commentaryBox.appendChild(row);
    commentaryBox.scrollTop = commentaryBox.scrollHeight;
    return { row, bubble };
  }

  function updateMessage(ref, role, text, opts = {}) {
    const { speaker, audioUrl } = opts;
    const row = ref.row;
    const bubble = ref.bubble;

    const meta = row.querySelector('.meta');
    if (meta) {
      const who = role === 'user' ? 'User' : (speaker || 'Model');
      meta.textContent = `${who} · ${nowCNTime()}`;
    }
    
     // Fix: use a strict check so 0 is not treated as false
    if (speaker !== undefined && speaker !== null) {
       const cls = 'speaker-' + String(speaker).toLowerCase().replace(/[^a-z0-9]/g, '-');
       // Prevent duplicate additions; clear old speaker classes first if needed
       // For simplicity, this assumes it changes only once or is only appended
       // For stricter handling, all speaker-* classes could be removed first, but updateMessage is usually one-time
       if (!row.classList.contains(cls)) {
         row.classList.add(cls);
       }
    }
    bubble.textContent = text;
    if (audioUrl) {
      const br = document.createElement('br');
      const tag = document.createElement('span');
      tag.className = 'audio-tag';
      tag.textContent = '🔊 Audio added to playback queue';
      tag.addEventListener('click', () => enqueueAudio(audioUrl));
      bubble.appendChild(br);
      bubble.appendChild(tag);
      enqueueAudio(audioUrl);
    }
  }

  function getAudioUrlFromData(data) {
    if (!data) return null;
    if (data.audio_url) return data.audio_url;
    if (data.audio_base64 && data.audio_mime) {
      return `data:${data.audio_mime};base64,${data.audio_base64}`;
    }
    return null;
  }

  // ========== Capture & frame extraction ==========
  const withStream = async (stream, kind) => {
    cleanupStream(); cleanupFile();
    mediaStream = stream; sourceKind  = kind;
    if (video) {
      video.srcObject = stream; video.muted = true;
      await video.play().catch(() => {});
    }
    enableSources(false); enableReadyControls(true);
    setStatus(kind === 'webcam' ? 'Webcam is ready' : 'Screen sharing is ready', true);
    if (kind === 'screen') {
      stream.getVideoTracks().forEach(tr => {
        tr.addEventListener('ended', () => { stopAll('Screen sharing ended'); });
      });
    }
  };

  const prepareWithStream = async (stream, kind) => {
    try { await withStream(stream, kind); } 
    catch (e) { derr(e); setStatus('Media stream initialization failed', false); }
  };

  async function startWebcam(){
    if (!navigator.mediaDevices?.getUserMedia){ setStatus('Webcam API is not supported', false); return; }
    try{
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } });
      await prepareWithStream(stream, 'webcam');
    } catch(e){ setStatus('Failed to access webcam', false); }
  }

  async function startScreen(){
    if (!navigator.mediaDevices?.getDisplayMedia){ setStatus('Screen sharing API is not supported', false); return; }
    try{
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
      await prepareWithStream(stream, 'screen');
    } catch(e){ setStatus('Screen sharing failed', false); }
  }

  function startFile(file){
    if (!file) return;
    cleanupStream(); cleanupFile();
    objectUrl = URL.createObjectURL(file);
    sourceKind = 'file';
    if (video) {
      video.srcObject = null; video.src = objectUrl;
      video.muted = true; video.loop  = true; video.controls = true;
      video.play().catch(() => {});
    }
    enableSources(false); enableReadyControls(true);
    setStatus(`Loaded local file: ${file.name}`, true);
  }

  function captureFrame(){
    if (!video || !video.videoWidth) return;
    const vw = video.videoWidth, vh = video.videoHeight;
    const targetW = Math.min(960, vw);
    const targetH = Math.round(vh * (targetW / vw));
    canvas.width = targetW; canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, targetW, targetH);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const fr = new FileReader();
      fr.onloadend = () => {
        const s = String(fr.result || '');
        const base64 = s.includes(',') ? s.split(',')[1] : s;
        lastFrames.push({ ts: Date.now(), w: targetW, h: targetH, mime: JPEG_MIME, data: base64 });
        if (lastFrames.length > 2) lastFrames = lastFrames.slice(-2);
      };
      fr.readAsDataURL(blob);
    }, JPEG_MIME, JPEG_QUALITY);
  }

  // ========== Logic update: show Assistant config blocks and send request ==========
  function showAssistantConfig() {
    let selectedCount = 1;
    if (assistantCountSelect) {
      selectedCount = parseInt(assistantCountSelect.value, 10);
    }
    
    dlog('[Settings] Setting assistant count to:', selectedCount);

    // 1. Update the UI
    if (assistantsConfigWrapper) assistantsConfigWrapper.style.display = 'block';
    assistantBlocks.forEach((block, index) => {
      const i = index + 1;
      if (block) block.style.display = (i <= selectedCount) ? 'block' : 'none';
    });
    setSettingsEnabled(!captureActive);

    // 2. Send a request to the backend: initialize N Assistants
    sendControl(
      'set_assistant_count', 
      { count: selectedCount }, 
      `Requested assistant count: ${selectedCount}`
    );
  }

  // ========== WebSocket ==========
  function ensureWS(){
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    ws = new WebSocket(BACKEND_WS_URL);
    ws.addEventListener('open', () => {
      setStatus('Connected to backend', true);
      ws.send(JSON.stringify({ type: 'hello', frames_per_sec: 2, want_comment_hz: 1 }));
      if (lastFrames.length && captureActive) sendInference();
    });
    ws.addEventListener('close', () => setStatus('Backend connection closed'));
    ws.addEventListener('error', () => setStatus('Backend connection error', false));
    ws.addEventListener('message', (ev) => {
      let data;
      try { data = JSON.parse(ev.data); } catch { data = { type: 'comment', text: String(ev.data||'') }; }

      if (data.type === 'comment' || data.type === 'commentary') {
        const rid = data.request_id;
        const text = data.text ?? data.commentary ?? '';
        const speaker = data.speaker;
        const audioUrl = getAudioUrlFromData(data);
        if (rid && inflight.has(rid)) {
          updateMessage(inflight.get(rid), 'model', text, { speaker, audioUrl });
          inflight.delete(rid);
        } else {
          appendMessage('model', text, { speaker, audioUrl });
        }
        return;
      }

      if (data.type === 'status' || data.type === 'error') {
        const rid = data.request_id;
        if (rid && inflightCtl.has(rid)) {
          const info = inflightCtl.get(rid);
          if (data.type === 'status') setSettingsHint(info.okMsg || (data.text || 'Completed'), true);
          else setSettingsHint(`Failed: ${data.text || 'Unknown error'}`, false);
          inflightCtl.delete(rid);
        } else if (data.type === 'status') setStatus(data.text, true);
        else if (data.type === 'error') appendMessage('model', `❗Error: ${data.text}`);
      }
    });
  }

  function sendControl(type, body, okMsg){
    if (captureActive) { setSettingsHint('Commentary is running. Pause it before changing settings', false); return; }
    ensureWS();
    if (!ws || ws.readyState !== WebSocket.OPEN) { 
      // If it is still connecting, wait until the connection opens before sending
      if (ws && ws.readyState === WebSocket.CONNECTING) {
         const tempReqId = ++reqSeq;
         inflightCtl.set(tempReqId, { okMsg });
         setSettingsHint('Applying...', true);
         const onOpen = () => {
            ws.removeEventListener('open', onOpen);
            const payload = { type, request_id: tempReqId, ...body };
            ws.send(JSON.stringify(payload));
         };
         ws.addEventListener('open', onOpen);
         return;
      }
      setSettingsHint('Backend is not connected', false); 
      return; 
    }
    
    const request_id = ++reqSeq;
    const payload = { type, request_id, ...body };
    inflightCtl.set(request_id, { okMsg });
    setSettingsHint('Applying...', true);
    ws.send(JSON.stringify(payload));
  }

  function sendInference(){
    if (!captureActive) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) { ensureWS(); return; }
    const frames = lastFrames.slice(-2);
    if (frames.length === 0) return;
    if (frames.length === 1) frames.unshift(frames[0]);

    const request_id = ++reqSeq;
    const payload = {
      type: 'infer', request_id, ts: Date.now(),
      source: sourceKind || 'unknown', frames, query: pendingQuery || undefined
    };
    pendingQuery = null;
    const ph = appendMessage('model', '(Analyzing...)');
    inflight.set(request_id, ph);
    ws.send(JSON.stringify(payload));
  }

  function startCapture(){
    if (captureActive) return;
    captureActive = true;
    setToggleUI(); ensureWS();
    captureTimer = setInterval(captureFrame, CAPTURE_INTERVAL_MS);
    inferTimer   = setInterval(sendInference,   INFER_INTERVAL_MS);
    setSettingsEnabled(false);
  }
  
  function pauseCapture(){
    if (!captureActive) return;
    captureActive = false;
    setToggleUI();
    if (captureTimer) { clearInterval(captureTimer); captureTimer = null; }
    if (inferTimer)   { clearInterval(inferTimer);   inferTimer   = null; }
    setSettingsEnabled(true);
  }

  function toggleCapture(){
    if (!captureActive) startCapture();
    else pauseCapture();
  }

  queryForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = (queryInput?.value || '').trim();
    if (!text) return;
    appendMessage('user', text);
    pendingQuery = text;
    queryInput.value = '';
  });

  // ========== Logic update: bind Assistant controls (Loop 1..3) ==========
  // Iterate through ID suffixes 1, 2, 3
  [1, 2, 3].forEach((idSuffix) => {
    // The backend assistant_id is 0-based (0, 1, 2)
    const assistantId = idSuffix - 1; 

    const rangeEl = document.getElementById(`thresholdRange-${idSuffix}`);
    const inputEl = document.getElementById(`thresholdInput-${idSuffix}`);
    const applyThreshBtn = document.getElementById(`applyThresholdBtn-${idSuffix}`);
    const clearCacheBtn = document.getElementById(`clearCacheBtn-${idSuffix}`);
    const promptEl = document.getElementById(`systemPrompt-${idSuffix}`);
    const applyPromptBtn = document.getElementById(`applySystemPromptBtn-${idSuffix}`);

    // 1. Sync threshold inputs
    if (rangeEl && inputEl) {
      rangeEl.addEventListener('input', () => {
        const v = clamp01(rangeEl.value);
        inputEl.value = v.toFixed(2);
      });
      inputEl.addEventListener('change', () => {
        const v = clamp01(inputEl.value);
        inputEl.value = v.toFixed(2);
        rangeEl.value = String(v);
      });
    }

    // 2. Apply threshold -> pass assistant_id
    if (applyThreshBtn) {
      applyThreshBtn.addEventListener('click', () => {
        const v = clamp01(inputEl?.value ?? rangeEl?.value ?? 0.5);
        if (inputEl) inputEl.value = v.toFixed(2);
        if (rangeEl) rangeEl.value = String(v);
        
        sendControl(
          'set_params', 
          { threshold: v, assistant_id: assistantId }, 
          `Assistant ${idSuffix}: Threshold applied: ${v.toFixed(2)}`
        );
      });
    }

    // 3. Clear cache -> pass assistant_id
    if (clearCacheBtn) {
      clearCacheBtn.addEventListener('click', () => {
        sendControl(
          'clear_cache', 
          { assistant_id: assistantId }, 
          `Assistant ${idSuffix}: Cache cleared`
        );
      });
    }

    // 4. Apply system prompt -> pass assistant_id
    if (applyPromptBtn) {
      applyPromptBtn.addEventListener('click', () => {
        const prompt = (promptEl?.value || '').trim();
        sendControl(
          'set_system_prompt', 
          { system_prompt: prompt, assistant_id: assistantId }, 
          `Assistant ${idSuffix}: System prompt updated`
        );
      });
    }
  });

  // ========== Event bindings ==========
  startWebcamBtn?.addEventListener('click', startWebcam);
  startScreenBtn?.addEventListener('click', startScreen);
  selectFileBtn?.addEventListener('click', () => fileInput?.click());
  fileInput?.addEventListener('change', (ev) => {
    const f = ev.target.files?.[0];
    if (f) startFile(f);
  });

  toggleCommentBtn?.addEventListener('click', () => toggleCapture());
  clearCommentaryBtn?.addEventListener('click', () => clearCommentaryBox());
  stopBtn?.addEventListener('click', () => stopAll());

  applyCountBtn?.addEventListener('click', showAssistantConfig);

  // ========== Initialization ==========
  enableSources(true);
  enableReadyControls(false);
  setToggleUI();              
  setSettingsEnabled(true);   
  setStatus('Disconnected');

  window.addEventListener('beforeunload', () => {
    try { ws && ws.close(); } catch {}
    stopAll();
  });
})();