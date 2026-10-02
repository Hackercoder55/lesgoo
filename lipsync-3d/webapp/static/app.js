// Lip-Sync Studio front end. Plain JS, no build step.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const state = {kind: 'clip', video: null, audio: null, faces: null, pick: null,
               point: null, img: null, me: null};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (r.status === 401) { location.href = '/login'; throw new Error('login'); }
  const body = r.headers.get('content-type')?.includes('json') ? await r.json() : null;
  if (!r.ok) {
    let d = body?.detail;
    if (Array.isArray(d)) d = d.map((x) => `${x.loc?.slice(-1)}: ${x.msg}`).join('\n');
    throw new Error(d || r.statusText);
  }
  return body;
}
const post = (p, b) => api(p, {method: 'POST', headers: {'content-type': 'application/json'},
                               body: JSON.stringify(b ?? {})});

// ---------------------------------------------------------------- header
async function loadMe() {
  const me = state.me = await api('/v1/me');
  $('#engine').textContent = me.engine === 'preview' ? 'engine: preview (no model)'
                                                     : `engine: LatentSync ${me.model || ''}`;
  $('#gpu').textContent = me.gpu_busy ? 'GPU busy' : 'GPU idle';
  $('#gpu').className = 'badge ' + (me.gpu_busy ? 'busy' : 'live');
  $('#who').textContent = me.local ? 'local mode' : me.name;
  $('#btnOut').hidden = $('#btnPw').hidden = me.local;
  $('#btnUsers').hidden = !me.admin || me.local;
  $('#allWrap').hidden = !me.admin || me.local;
  $('#warn').innerHTML = (me.missing || []).map((m) =>
    `<div class="warnbox">Setup needed: ${esc(m)}</div>`).join('');
  if (me.vram_gb && !state.vramSet) {
    state.vramSet = true;
    $('#gpu').title = `${me.vram_gb} GB GPU`;
    if (me.vram_gb < 10) {
      // small GPU: no classifier-free guidance halves the model's memory,
      // and a smaller crop keeps the frames small
      $('#guidance').value = '1.0';
      $('#cropMax').value = '512';
      $('#engine').textContent += ` · low-VRAM settings (${me.vram_gb} GB GPU)`;
    }
  }
}
$('#btnOut').onclick = async () => { await post('/auth/logout'); location.href = '/login'; };

// ------------------------------------------------------------------ tabs
const hints = {
  clip: 'One shot: pick the face, optionally give new audio, get it back lip-synced.',
  auto: 'A whole video with up to 4 characters: detect the characters once, switch on the ones ' +
        'to lip-sync, and each is synced wherever its mouth moves during dialogue. Everything else stays untouched.',
};
document.querySelectorAll('.tabs button').forEach((b) => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
  state.kind = b.dataset.kind;
  $('#clipOnly').hidden = state.kind !== 'clip';
  $('#autoOnly').hidden = state.kind !== 'auto';
  $('#autoPlanBox').hidden = state.kind !== 'auto';
  $('#kindHint').textContent = hints[state.kind];
});
$('#kindHint').textContent = hints.clip;

// --------------------------------------------------------------- uploads
function upload(file, drop) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append('file', file);
    const x = new XMLHttpRequest();
    x.open('POST', '/v1/assets');
    x.upload.onprogress = (e) => {
      drop.querySelector('.bar i').style.width = `${(e.loaded / e.total) * 100}%`;
    };
    x.onload = () => {
      let b = {};
      try { b = JSON.parse(x.responseText); } catch (e) { /* not json */ }
      if (x.status === 200) resolve(b); else reject(new Error(b.detail || x.statusText));
    };
    x.onerror = () => reject(new Error('upload failed - is the server running?'));
    x.send(fd);
  });
}

function setupDrop(drop, slot, onDone) {
  const input = drop.querySelector('input');
  const msg = drop.querySelector('.msg');
  const go = async (file) => {
    if (!file) return;
    $('#err').textContent = '';
    msg.innerHTML = `Uploading <span class="name">${esc(file.name)}</span>…`;
    drop.querySelector('.bar i').style.width = '0';
    try {
      const a = await upload(file, drop);
      state[slot] = a;
      const dims = a.w ? ` · ${a.w}×${a.h} · ${a.fps.toFixed(2)} fps` : '';
      msg.innerHTML = `<span class="name">${esc(a.name)}</span><br>${a.duration.toFixed(1)}s${dims} · click to replace`;
      onDone && onDone(a);
    } catch (e) {
      state[slot] = null;
      msg.textContent = 'Upload failed: ' + e.message;
    }
    refreshGo();
  };
  drop.onclick = (e) => { if (e.target !== input) input.click(); };
  input.onchange = () => go(input.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over');
                         go(e.dataTransfer.files[0]); };
}
setupDrop($('#dropVideo'), 'video', (a) => {
  if (a.kind !== 'video') { state.video = null; $('#err').textContent = 'That file has no video.'; }
  state.faces = state.pick = state.point = null;
  $('#picker').hidden = true; $('#faceList').innerHTML = ''; $('#manualSize').hidden = true;
  $('#btnFaces').disabled = !state.video;
  $('#btnAnalyze').disabled = !state.video;
  state.plan = null; $('#plan').innerHTML = ''; $('#chars').innerHTML = ''; $('#tlWrap').hidden = true; $('#advParts').hidden = true;
  $('#faceT').value = Math.min(0.5, (a.duration || 1) / 2).toFixed(1);
  // show the frame and its faces straight away in clip mode
  if (state.video && state.kind === 'clip') $('#btnFaces').click();
});
setupDrop($('#dropAudio'), 'audio');

function refreshGo() { $('#btnGo').disabled = !state.video; }

// ------------------------------------------------------------ face pick
$('#btnFaces').onclick = async () => {
  if (!state.video) return;
  const t = parseFloat($('#faceT').value) || 0;
  const score = parseFloat($('#score').value) || 0.5;
  $('#btnFaces').disabled = true; $('#btnFaces').textContent = 'Looking…';
  try {
    const [f, img] = await Promise.all([
      api(`/v1/assets/${state.video.id}/faces?t=${t}&score=${score}`),
      new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i);
        i.onerror = () => rej(new Error('could not read that frame'));
        i.src = `/v1/assets/${state.video.id}/frame?t=${t}&_=${Date.now()}`; }),
    ]);
    state.faces = f; state.img = img; state.point = null;
    state.pick = f.faces.length ? 0 : null;
    $('#picker').hidden = false;
    $('#faceHint').textContent = f.faces.length
      ? `Found ${f.faces.length} face(s). Face 1 (most confident) is selected — click another to change.`
      : 'No face found on this frame. Click on the character\'s face, then set its size below.';
    drawFaces();
  } catch (e) { $('#err').textContent = e.message; }
  $('#btnFaces').disabled = false; $('#btnFaces').textContent = 'Find faces at this second';
};

function manualBox() {
  const h = state.faces.height * (+$('#size').value) / 100;
  return [state.point[0], state.point[1], h * 0.8, h];
}

function drawFaces() {
  const cv = $('#cv'), f = state.faces;
  cv.width = f.width; cv.height = f.height;
  const g = cv.getContext('2d');
  g.drawImage(state.img, 0, 0, f.width, f.height);
  const lw = Math.max(3, f.height / 250);
  g.lineWidth = lw; g.font = `bold ${lw * 8}px sans-serif`;
  f.faces.forEach((x, i) => {
    const [a, b, w, h] = x.box;
    g.strokeStyle = g.fillStyle = state.pick === i ? '#3ccf8e' : '#f5b942';
    g.strokeRect(a, b, w, h);
    g.fillText(String(i + 1), a, b - lw * 2);
  });
  if (state.point) {
    const [cx, cy, w, h] = manualBox();
    g.strokeStyle = '#3ccf8e'; g.strokeRect(cx - w / 2, cy - h / 2, w, h);
    g.strokeStyle = '#ff5d6c'; g.beginPath();
    g.moveTo(cx - 20, cy); g.lineTo(cx + 20, cy); g.moveTo(cx, cy - 20); g.lineTo(cx, cy + 20);
    g.stroke();
  }
  $('#faceList').innerHTML = f.faces.map((x, i) =>
    `<button class="small ${state.pick === i ? 'on' : ''}" data-i="${i}">Face ${i + 1} · ${Math.round(x.score * 100)}%</button>`
  ).join('');
  document.querySelectorAll('#faceList button').forEach((b) => b.onclick = () => {
    state.pick = +b.dataset.i; state.point = null; $('#manualSize').hidden = true; drawFaces();
  });
}

$('#cv').onclick = (e) => {
  const cv = $('#cv'), r = cv.getBoundingClientRect();
  const x = (e.clientX - r.left) * cv.width / r.width;
  const y = (e.clientY - r.top) * cv.height / r.height;
  const hit = state.faces.faces.findIndex(({box: [a, b, w, h]}) =>
    x >= a && x <= a + w && y >= b && y <= b + h);
  if (hit >= 0) { state.pick = hit; state.point = null; $('#manualSize').hidden = true; }
  else { state.pick = null; state.point = [x, y]; $('#manualSize').hidden = false; }
  drawFaces();
};
$('#size').oninput = () => { $('#sizeVal').textContent = $('#size').value;
                             if (state.point) drawFaces(); };

// ---------------------------------------------------------------- submit
$('#btnGo').onclick = async () => {
  $('#err').textContent = '';
  const num = (id) => parseFloat($(id).value);
  const b = {kind: state.kind, video: state.video.id, steps: num('#steps'),
             guidance: num('#guidance'), seed: num('#seed'), crop_max: num('#cropMax'),
             score: num('#score'), mask_scale: num('#maskScale')};
  if (state.kind === 'clip') {
    b.mode = $('#mode').value;
    if (state.audio) b.audio = state.audio.id;
    if (state.faces && state.pick !== null) {
      const [a, y, w, h] = state.faces.faces[state.pick].box;
      b.face = {t: state.faces.t, box: [a + w / 2, y + h / 2, w, h]};
    } else if (state.faces && state.point) {
      b.face = {t: state.faces.t, box: manualBox()};
    }
  } else {
    b.auto = autoOpts();
    if (state.plan) {
      b.plan = state.plan.plan;
      b.picks = picks();
      if (!Object.values(b.picks).some((x) => x.length)) {
        $('#err').textContent = 'Nothing would be lip-synced - switch on a character (or force a face on under "Fix a part by hand").';
        return;
      }
    }
  }
  $('#btnGo').disabled = true;
  try { await post('/v1/jobs', b); await loadJobs(); }
  catch (e) { $('#err').textContent = e.message; }
  refreshGo();
};

// ------------------------------------------------------- auto-mode plan
const COLORS = {c1: '#ff6a2b', c2: '#5ea8ff', c3: '#3ccf8e', c4: '#f5b942', other: '#9298a8'};
function autoOpts() {
  const num = (id) => parseFloat($(id).value);
  return {threshold_db: num('#thr'), merge_gap: num('#gap'), pad_before: num('#padB'),
          pad_after: num('#padA'), scene_cut: num('#cut'),
          characters: parseInt($('#nChars').value), mouth_threshold: num('#mouthSens')};
}
const fmt = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`;

// a face is synced when forced on, or when its character is on and its mouth moves
function faceOn(s, f) {
  const k = `${s.id}/${f.id}`;
  if (k in state.plan.force) return state.plan.force[k];
  return !!(state.plan.on[f.char] && f.moving.length);
}
function picks() {
  return Object.fromEntries(state.plan.segments.map((s) =>
    [s.id, s.faces.filter((f) => faceOn(s, f)).map((f) => f.id)]));
}

$('#btnAnalyze').onclick = async () => {
  if (!state.video) return;
  $('#err').textContent = '';
  $('#btnAnalyze').disabled = true;
  $('#btnAnalyze').textContent = 'Detecting characters… (about a minute per few minutes of video)';
  try {
    const p = await post(`/v1/assets/${state.video.id}/analyze`,
                         {score: parseFloat($('#score').value) || 0.5, auto: autoOpts()});
    p.on = Object.fromEntries(p.characters.map((c) => [c.id, c.id !== 'other']));
    p.names = Object.fromEntries(p.characters.map((c) => [c.id, c.name]));
    p.force = {};
    state.plan = p;
    drawPlan();
  } catch (e) { $('#err').textContent = e.message; }
  $('#btnAnalyze').disabled = false;
  $('#btnAnalyze').textContent = 'Detect characters again';
};

function drawPlan() {
  const p = state.plan;
  const faceByKey = {};
  p.segments.forEach((s) => s.faces.forEach((f) => { faceByKey[`${s.id}/${f.id}`] = [s, f]; }));
  // character cards
  $('#chars').innerHTML = p.characters.length ? p.characters.map((c) => {
    const mine = c.tracks.map((k) => faceByKey[k]).filter(Boolean);
    const synced = mine.filter(([s, f]) => faceOn(s, f));
    const talk = synced.reduce((t, [, f]) => t + (f.moving.length ? f.moving_s : f.visible_s), 0);
    const pics = c.tracks.slice(0, 5).map((k) => faceByKey[k]?.[1].thumb).filter(Boolean)
      .map((u) => `<img src="${u}" loading="lazy" alt="">`).join('');
    return `<div class="char ${p.on[c.id] ? '' : 'off'}" style="--c:${COLORS[c.id] || '#999'}" data-c="${c.id}">
      <div class="top"><input value="${esc(p.names[c.id])}" data-name="${c.id}" title="rename">
        <button class="tog ${p.on[c.id] ? 'on' : ''}" data-tog="${c.id}">${p.on[c.id] ? 'Lip-sync ON' : 'Lip-sync OFF'}</button></div>
      <div class="pics">${pics}</div>
      <div class="stat">${c.tracks.length} shot(s)${c.visible_s ? ` · on screen ${c.visible_s}s` : ''} · mouth moves ${c.moving_s}s · <b>will sync ${talk.toFixed(1)}s in ${synced.length} part(s)</b></div>
    </div>`;
  }).join('') : '<div class="empty">No faces found. Lower "Detector confidence" in Settings and detect again.</div>';
  document.querySelectorAll('[data-tog]').forEach((b) => b.onclick = () => {
    p.on[b.dataset.tog] = !p.on[b.dataset.tog]; drawPlan();
  });
  document.querySelectorAll('[data-name]').forEach((i) => i.oninput = () => {
    p.names[i.dataset.name] = i.value; drawTimeline();
  });
  const total = Object.values(picks()).reduce((n, x) => n + x.length, 0);
  $('#planHint').textContent = p.characters.length
    ? `${p.characters.filter((c) => c.id !== 'other').length} character(s) found in ${p.segments.length} speaking part(s). ` +
      `${total} face-part(s) will be lip-synced. Switch characters on/off; check the timeline.`
    : 'No characters found.';
  $('#tlWrap').hidden = !p.characters.length;
  $('#advParts').hidden = !p.segments.length;
  drawTimeline();
  // advanced per-part list
  $('#plan').innerHTML = p.segments.map((s) => {
    const n = s.faces.filter((f) => faceOn(s, f)).length;
    return `<div class="seg" data-s="${s.id}">
      <div class="when"><span>${fmt(s.start)} – ${fmt(s.end)}</span><span>${n ? n + ' synced' : 'not synced'}</span></div>
      ${s.faces.length ? `<div class="faces">${s.faces.map((f) => {
        const on = faceOn(s, f), forced = `${s.id}/${f.id}` in p.force;
        return `<button class="face ${on ? 'on' : ''}" data-f="${f.id}" style="--c:${COLORS[f.char] || '#999'}"
          title="${f.moving.length ? 'mouth moves ' + f.moving_s + 's' : 'mouth does not move'}${forced ? ' (set by hand)' : ''}">
          <img src="${f.thumb}" loading="lazy" alt=""><i class="who">${esc(p.names[f.char] || '?')}</i>
          <span>${on ? '✓ sync' : 'skip'}${forced ? ' ✎' : ''}</span></button>`;
      }).join('')}</div>` : '<div class="none">No face found here.</div>'}
    </div>`;
  }).join('');
}

function drawTimeline() {
  const p = state.plan, cv = $('#tl');
  const chars = p.characters;
  const rowH = 22, top = 18, W = cv.clientWidth || 480, H = top + rowH * chars.length + 6;
  cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio; cv.style.height = H + 'px';
  const g = cv.getContext('2d');
  g.scale(devicePixelRatio, devicePixelRatio);
  g.clearRect(0, 0, W, H);
  const L = 90, x = (t) => L + (W - L - 8) * t / p.duration;
  g.font = '10px sans-serif'; g.fillStyle = '#9298a8';
  const step = p.duration > 600 ? 60 : p.duration > 120 ? 30 : p.duration > 30 ? 10 : 5;
  for (let t = 0; t <= p.duration; t += step) { g.fillText(fmt(t).replace(/\.\d$/, ''), x(t) - 8, 11); g.fillRect(x(t), 14, 1, H - 14); }
  chars.forEach((c, r) => {
    const y = top + r * rowH;
    g.fillStyle = COLORS[c.id] || '#999'; g.font = '11px sans-serif';
    g.fillText((p.names[c.id] || c.id).slice(0, 13), 4, y + 14);
    c.tracks.forEach((k) => {
      const [sid, fid] = k.split('/');
      const s = p.segments.find((z) => z.id === sid), f = s?.faces.find((z) => z.id === fid);
      if (!f) return;
      g.fillStyle = '#2f3442'; g.fillRect(x(f.seen[0]), y + 4, Math.max(1, x(f.seen[1]) - x(f.seen[0])), rowH - 8);
      if (faceOn(s, f)) {
        g.fillStyle = COLORS[c.id] || '#999';
        (f.moving.length ? f.moving : [f.seen]).forEach(([a, b]) =>
          g.fillRect(x(a), y + 4, Math.max(2, x(b) - x(a)), rowH - 8));
      }
    });
  });
}
window.addEventListener('resize', () => state.plan && drawTimeline());

$('#plan').onclick = (e) => {
  const b = e.target.closest('.face');
  if (!b) return;
  const s = state.plan.segments.find((x) => x.id === b.closest('.seg').dataset.s);
  const f = s.faces.find((x) => x.id === b.dataset.f);
  state.plan.force[`${s.id}/${f.id}`] = !faceOn(s, f);
  drawPlan();
};

// ------------------------------------------------------------------ jobs
const ago = (t) => {
  const s = Date.now() / 1000 - t;
  return s < 60 ? 'just now' : s < 3600 ? `${Math.floor(s / 60)} min ago`
       : s < 86400 ? `${Math.floor(s / 3600)} h ago` : new Date(t * 1000).toLocaleString();
};
async function loadJobs() {
  const all = $('#all').checked ? '&all=true' : '';
  const jobs = await api(`/v1/jobs?limit=100${all}`);
  $('#jobs').innerHTML = jobs.length ? jobs.map((j) => {
    const st = j.status === 'queued' && j.queue_ahead ? `queued · ${j.queue_ahead} ahead` : j.status;
    const acts = [
      `<button class="small" data-act="view">View</button>`,
      j.result_url ? `<a href="${j.result_url}"><button class="small">Download</button></a>` : '',
      ['queued', 'running'].includes(j.status) ? `<button class="small" data-act="cancel">Cancel</button>` : '',
      ['failed', 'cancelled', 'done'].includes(j.status) ? `<button class="small" data-act="retry">Run again</button>` : '',
      !['running', 'cancelling'].includes(j.status) ? `<button class="small danger" data-act="delete">Delete</button>` : '',
    ].join('');
    return `<div class="job" data-id="${j.id}">
      <div><div class="title"><span class="pill ${j.status}">${esc(st)}</span>${esc(j.video_name)}</div>
      <div class="meta">${j.kind === 'auto' ? 'Full video (auto)' : 'Clip'} · ${ago(j.created)}${j.seconds ? ` · took ${Math.round(j.seconds)}s` : ''}${j.owner ? ` · ${esc(j.owner)}` : ''}</div></div>
      <div class="acts">${acts}</div>
      <div class="last">${esc(j.error || j.log || '')}</div></div>`;
  }).join('') : '<div class="empty">No jobs yet.</div>';
}
$('#all').onchange = loadJobs;
$('#jobs').onclick = async (e) => {
  const b = e.target.closest('button[data-act]');
  if (!b) return;
  const id = b.closest('.job').dataset.id, act = b.dataset.act;
  try {
    if (act === 'view') return showJob(id);
    if (act === 'delete' && !confirm('Delete this job and its result?')) return;
    if (act === 'delete') await api(`/v1/jobs/${id}`, {method: 'DELETE'});
    else await post(`/v1/jobs/${id}/${act}`);
    loadJobs();
  } catch (err) { alert(err.message); }
};

// ---------------------------------------------------------------- modals
function modal(html) { $('#mbox').innerHTML = html; $('#modal').classList.add('open'); }
function closeModal() { $('#modal').classList.remove('open'); $('#mbox').innerHTML = ''; }
$('#modal').onclick = (e) => { if (e.target.id === 'modal' || e.target.dataset.close !== undefined) closeModal(); };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
const head = (t) => `<h3>${t}<button class="small ghost" data-close>✕</button></h3>`;

async function showJob(id) {
  const j = await api(`/v1/jobs/${id}`);
  const rep = j.report ? `<h4>Segments</h4><table><tr><th>Segment</th><th>Time</th><th>Faces</th><th>Status</th><th></th></tr>${
    j.report.map((r) => `<tr><td>${r.segment}</td><td>${r.start.toFixed(2)}–${r.end.toFixed(2)}s</td>
      <td>${esc((r.faces || []).join(', '))}</td>
      <td><span class="pill ${r.status === 'synced' ? 'done' : r.status === 'failed' ? 'failed' : r.status === 'partly synced' ? 'running' : 'cancelled'}">${r.status}</span></td>
      <td>${esc(r.why || '')}</td></tr>`).join('')}</table>` : '';
  modal(head(esc(j.video_name)) +
    (j.result_url ? `<video src="${j.result_url}" controls autoplay></video>
      <p><a href="${j.result_url}"><button class="primary">Download</button></a></p>` : '') +
    (j.error ? `<div class="err">${esc(j.error)}</div>` : '') + rep +
    `<h4>Log</h4><pre class="log">${esc(j.log || '(nothing yet)')}</pre>`);
}

$('#btnKeys').onclick = async () => {
  const keys = await api('/v1/keys');
  modal(head('API keys') +
    `<p class="hint">Use a key in the <code>x-api-key</code> header. Docs: <a href="/v1/docs" target="_blank">/v1/docs</a></p>
     <div id="newKey"></div>
     <button class="primary" id="mkKey">Create key</button>
     <table style="margin-top:12px"><tr><th>Key</th><th>Created</th><th></th></tr>${
       keys.map((k) => `<tr><td><code>${esc(k.prefix)}…</code></td><td>${ago(k.created)}</td>
       <td><button class="small danger" data-del="${k.id}">Revoke</button></td></tr>`).join('')}</table>
     <h4>Example</h4><pre class="log">KEY=ls_...
curl -H "x-api-key: $KEY" -F file=@clip.mp4 ${location.origin}/v1/assets
curl -H "x-api-key: $KEY" -H "content-type: application/json" \\
     -d '{"kind":"clip","video":"ASSET_ID"}' ${location.origin}/v1/jobs
curl -H "x-api-key: $KEY" ${location.origin}/v1/jobs/JOB_ID
curl -H "x-api-key: $KEY" -o out.mp4 ${location.origin}/v1/jobs/JOB_ID/result</pre>`);
  $('#mkKey').onclick = async () => {
    const k = await post('/v1/keys');
    $('#newKey').innerHTML = `<p>Copy it now — it will not be shown again:</p><code class="key">${esc(k.key)}</code>`;
  };
  document.querySelectorAll('[data-del]').forEach((b) => b.onclick = async () => {
    await api(`/v1/keys/${b.dataset.del}`, {method: 'DELETE'}); $('#btnKeys').click();
  });
};

$('#btnUsers').onclick = async () => {
  const users = await api('/v1/users');
  modal(head('Users') +
    `<table><tr><th>Name</th><th>Role</th><th>Jobs</th><th></th></tr>${
      users.filter((u) => u.name !== 'local').map((u) => `<tr><td>${esc(u.name)}</td><td>${u.admin ? 'admin' : 'member'}</td>
      <td>${u.jobs}</td><td>${u.name === state.me.name ? '' : `<button class="small danger" data-rm="${u.id}">Remove</button>`}</td></tr>`).join('')}</table>
     <h4>Add a user</h4>
     <div class="grid2"><div><label>Name</label><input id="nuName"></div>
     <div><label>Password (8+ characters)</label><input id="nuPw" type="text"></div></div>
     <label><input type="checkbox" id="nuAdmin" style="width:auto"> admin</label>
     <button class="primary" id="nuGo">Add</button><div class="err" id="nuErr"></div>`);
  $('#nuGo').onclick = async () => {
    try {
      await post('/v1/users', {name: $('#nuName').value, password: $('#nuPw').value,
                               admin: $('#nuAdmin').checked});
      $('#btnUsers').click();
    } catch (e) { $('#nuErr').textContent = e.message; }
  };
  document.querySelectorAll('[data-rm]').forEach((b) => b.onclick = async () => {
    if (!confirm('Remove this user?')) return;
    await api(`/v1/users/${b.dataset.rm}`, {method: 'DELETE'}); $('#btnUsers').click();
  });
};

$('#btnPw').onclick = () => {
  modal(head('Change password') +
    `<label>Current password</label><input id="pwOld" type="password">
     <label>New password (8+ characters)</label><input id="pwNew" type="password">
     <p><button class="primary" id="pwGo">Save</button></p><div class="err" id="pwErr"></div>`);
  $('#pwGo').onclick = async () => {
    try { await post('/v1/me/password', {old: $('#pwOld').value, new: $('#pwNew').value});
          closeModal(); }
    catch (e) { $('#pwErr').textContent = e.message; }
  };
};

// ------------------------------------------------------------------ boot
(async () => {
  await loadMe();
  await loadJobs();
  setInterval(() => { loadJobs().catch(() => {}); loadMe().catch(() => {}); }, 3000);
})();
