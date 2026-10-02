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
                                                     : 'engine: LatentSync';
  $('#gpu').textContent = me.gpu_busy ? 'GPU busy' : 'GPU idle';
  $('#gpu').className = 'badge ' + (me.gpu_busy ? 'busy' : 'live');
  $('#who').textContent = me.local ? 'local mode' : me.name;
  $('#btnOut').hidden = $('#btnPw').hidden = me.local;
  $('#btnUsers').hidden = !me.admin || me.local;
  $('#allWrap').hidden = !me.admin || me.local;
  $('#warn').innerHTML = (me.missing || []).map((m) =>
    `<div class="warnbox">Setup needed: ${esc(m)}</div>`).join('');
}
$('#btnOut').onclick = async () => { await post('/auth/logout'); location.href = '/login'; };

// ------------------------------------------------------------------ tabs
const hints = {
  clip: 'One shot: pick the face, optionally give new audio, get it back lip-synced.',
  auto: 'A whole episode: speech is found automatically and only those parts are ' +
        're-synced to the video\'s own dialogue. Everything else stays untouched.',
};
document.querySelectorAll('.tabs button').forEach((b) => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
  state.kind = b.dataset.kind;
  $('#clipOnly').hidden = state.kind !== 'clip';
  $('#autoOnly').hidden = state.kind !== 'auto';
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
  $('#faceT').value = Math.min(0.5, (a.duration || 1) / 2).toFixed(1);
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
             score: num('#score')};
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
    b.auto = {threshold_db: num('#thr'), merge_gap: num('#gap'),
              pad_before: num('#padB'), pad_after: num('#padA')};
  }
  $('#btnGo').disabled = true;
  try { await post('/v1/jobs', b); await loadJobs(); }
  catch (e) { $('#err').textContent = e.message; }
  refreshGo();
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
  const rep = j.report ? `<h4>Segments</h4><table><tr><th>Segment</th><th>Time</th><th>Status</th><th></th></tr>${
    j.report.map((r) => `<tr><td>${r.segment}</td><td>${r.start.toFixed(2)}–${r.end.toFixed(2)}s</td>
      <td><span class="pill ${r.status === 'synced' ? 'done' : r.status === 'failed' ? 'failed' : 'cancelled'}">${r.status}</span></td>
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
