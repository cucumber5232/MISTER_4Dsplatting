/* MISTER 4D Studio: the page. It shows the state kept by the local server
   (app.py) and sends back what the operator chooses; it computes nothing. */
(() => {
  'use strict';

  const TOKEN = document.querySelector('meta[name="studio-token"]').content;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const STEPS = [
    { id: 'progetto', name: 'Progetto' },
    { id: 'video', name: 'Video' },
    { id: 'calibrazione', name: 'Calibrazione' },
    { id: 'preparazione', name: 'Preparazione' },
    { id: 'ricostruzione', name: 'Ricostruzione' },
    { id: 'risultato', name: 'Risultato' },
  ];
  const STATUS = { todo: 'Da fare', running: 'In corso', done: 'Fatto', error: 'Non riuscito' };
  const MARK = {
    done: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    error: '<svg viewBox="0 0 24 24"><path d="M12 6v8M12 18v.5"/></svg>',
  };
  const PICK_TITLE = {
    calib: 'Cartella con i video del pannello di calibrazione',
    empty: 'Cartella con i video del set vuoto',
    take: 'Cartella con i video della ripresa del soggetto',
  };

  let state = null;       // last state received from the server
  let current = null;     // the step on screen
  let follow = false;     // while "run the remaining steps" is going, show the step at work
  let openPath = null;    // project the inputs were filled for
  let closed = false;
  const dirty = new Set();  // fields being edited: the server must not overwrite them

  // ---------------------------------------------------------------- helpers

  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value == null || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const num = (value, digits = 0) => Number(value).toLocaleString('it-IT',
    { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const count = (n, one, many) => `${num(n)} ${n === 1 ? one : many}`;
  const signed = (value, digits = 0) => (value > 0 ? '+' : value < 0 ? '−' : '') + num(Math.abs(value), digits);
  const known = (value) => typeof value === 'number' && Number.isFinite(value);

  function duration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    if (s < 60) return `${s} s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m} min ${String(s % 60).padStart(2, '0')} s`;
    return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`;
  }

  let toastTimer = 0;
  function toast(message) {
    const box = $('#toast');
    box.textContent = message;
    box.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { box.hidden = true; }, 8000);
  }

  // ----------------------------------------------------------------- server

  async function request(path, body) {
    const init = { headers: { 'X-Studio-Token': TOKEN } };
    if (body !== undefined) {
      init.method = 'POST';
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    const response = await fetch(path, init);
    let data = null;
    try { data = await response.json(); } catch { /* not JSON: handled below */ }
    if (!response.ok) throw new Error((data && data.error) || 'Il programma non ha risposto come previsto.');
    return data;
  }

  // Changes go out one at a time, in the order they were made: an option typed
  // just before pressing a button reaches the server before the button does.
  let queue = Promise.resolve();
  let sequence = 0;
  let shown = 0;
  function send(path, body) {
    const result = queue.then(() => {
      const mine = ++sequence;
      return request(path, body).then((data) => {
        if (data && data.steps) accept(data, mine);
        return data;
      });
    });
    queue = result.catch(() => {});
    return result.catch((error) => { toast(error.message); return null; });
  }

  function accept(data, mine) {
    if (mine < shown) return;  // an older answer arriving late
    shown = mine;
    render(data);
    schedule();
  }

  // Read the state again: every second while something runs, otherwise now and then.
  function schedule() {
    clearTimeout(timer);
    if (!closed) timer = setTimeout(poll, state && state.job ? 1000 : 4000);
  }

  let timer = 0;
  let failures = 0;
  async function poll() {
    clearTimeout(timer);
    if (closed) return;
    const mine = ++sequence;
    try {
      accept(await request('/api/state'), mine);
      failures = 0;
    } catch {
      failures += 1;
      if (failures >= 3) {
        return goodbye('4D Studio non risponde',
          'Il programma è stato chiuso o si è fermato. Riaprilo con doppio clic su «MISTER 4D Studio», poi ricarica questa pagina.',
          true);
      }
    }
    schedule();
  }

  function goodbye(title, text, retry) {
    closed = true;
    clearTimeout(timer);
    if ($('.goodbye')) return;
    document.body.append(el('div', { class: 'goodbye', role: 'alertdialog', 'aria-labelledby': 'goodbyeTitle' },
      el('h2', { id: 'goodbyeTitle', text: title }), el('p', { text }),
      retry && el('div', { class: 'actions', style: 'justify-content:center' },
        el('button', { class: 'btn primary', type: 'button', text: 'Ricarica la pagina', onclick: () => location.reload() }))));
  }

  // ------------------------------------------------------------- navigation

  const backend = (id) => state.steps.find((step) => step.id === id);
  const sectionOf = (jobStep) => (jobStep === 'esempio' ? 'video' : jobStep);
  const nameOf = (id) => STEPS.find((step) => step.id === id).name;

  function statusOf(id) {
    if (id === 'progetto') return state.project ? 'done' : 'todo';
    if (state.job && sectionOf(state.job.step) === id) return 'running';
    return backend(id).status;
  }

  function defaultStep() {
    if (!state.project) return 'progetto';
    if (state.job) return sectionOf(state.job.step);
    const next = state.steps.find((step) => step.status !== 'done');
    return next ? next.id : 'risultato';
  }

  function go(id, focus) {
    current = id;
    show();
    $('#stage').scrollTop = 0;
    if (focus) $('#stage').focus({ preventScroll: true });
  }

  function show() {
    for (const section of $$('.step')) section.hidden = section.dataset.step !== current;
    for (const button of $$('.step-link')) {
      const here = button.dataset.go === current;
      if (here && !button.hasAttribute('aria-current')) {
        button.setAttribute('aria-current', 'step');
        const rail = $('#rail');  // on narrow screens the steps scroll sideways
        if (rail.scrollWidth > rail.clientWidth) rail.scrollLeft = button.offsetLeft - rail.offsetLeft - 12;
      } else if (!here) button.removeAttribute('aria-current');
    }
  }

  // ------------------------------------------------------------------ build

  function build() {
    const rail = $('#rail');
    STEPS.forEach((step, index) => {
      const button = el('button', { class: 'step-link', type: 'button', 'data-go': step.id },
        el('span', { class: 'num', text: String(index + 1).padStart(2, '0') }),
        el('span', {}, el('span', { class: 'name', text: step.name }), el('span', { class: 'state' })),
        el('span', { class: 'mark', 'aria-hidden': 'true' }));
      button.addEventListener('click', () => { follow = false; go(step.id); });
      rail.append(el('li', {}, button));
    });

    // Every step ends the same way: the action, then what came out of it,
    // then the technical log for whoever needs it.
    for (const section of $$('.step')) {
      const id = section.dataset.step;
      if (id === 'progetto') continue;
      const actions = $('.actions', section);
      const button = $('[data-run]', section);
      button.dataset.label = button.textContent;
      actions.append(el('span', { class: 'blocked-note' }));
      const fold = el('details', { class: 'fold tech', hidden: true },
        el('summary', { text: 'Dettagli tecnici' }), el('pre', { class: 'log', tabindex: '0' }));
      fold.addEventListener('toggle', () => { if (state) renderLog(backend(id), section); });
      actions.after(el('div', { class: 'outcome' }), fold);
    }
    $('#exampleBox').append(el('div', { class: 'outcome' }));
    $('[data-step="progetto"] .field').after(el('div', { class: 'actions', id: 'projectNext', hidden: true },
      el('button', { class: 'btn primary', type: 'button', text: 'Continua con i video', onclick: () => go('video', true) })));

    drawRig();
  }

  // The backdrop: eight cameras around the subject, four more raised and offset.
  function drawRig() {
    const group = $('#rigCameras');
    const add = (radius, angle, raised) => {
      const camera = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      camera.setAttribute('transform', `rotate(${angle} 390 390) translate(390 ${390 - radius})`);
      if (raised) camera.setAttribute('class', 'raised');
      camera.innerHTML = '<rect x="-17" y="-11" width="34" height="22" rx="4"/><path class="lens" d="M-7 11h14l4 9h-22z"/>';
      group.append(camera);
    };
    for (let i = 0; i < 8; i++) add(250, i * 45, false);
    for (let i = 0; i < 4; i++) add(318, 22.5 + i * 90, true);
  }

  // ----------------------------------------------------------------- render

  function render(data) {
    const before = state;
    state = data;
    const path = state.project ? state.project.path : null;
    if (path !== openPath) {  // another project: start from its own values
      openPath = path;
      dirty.clear();
      current = null;
    }
    if (current === null) current = defaultStep();
    if (follow && state.job && (!before || !before.job || before.job.step !== state.job.step)) {
      current = sectionOf(state.job.step);
    }
    if (!state.job) follow = false;

    renderBar();
    renderRail();
    renderProject();
    renderVideos();
    fillInputs();
    for (const step of state.steps) renderStep(step);
    renderExample();
    show();
  }

  function renderBar() {
    const name = $('#projectName');
    if (state.project) name.replaceChildren(el('b', { text: state.project.name }), ` · ${state.project.path}`);
    else name.textContent = 'Nessun progetto aperto';
    $('#openProject').hidden = !state.project;
  }

  function renderRail() {
    for (const button of $$('.step-link')) {
      const status = statusOf(button.dataset.go);
      if (button.dataset.status !== status) {
        button.dataset.status = status;
        $('.state', button).textContent = STATUS[status];
        $('.mark', button).innerHTML = MARK[status] || '';
      }
    }
    const rest = state.steps.slice(Math.max(0, state.steps.findIndex((step) => step.status !== 'done')));
    const allDone = state.steps.every((step) => step.status === 'done');
    let reason = '';
    if (!state.project) reason = 'Apri prima un progetto.';
    else if (state.job) reason = 'Un passo è in corso.';
    else if (allDone) reason = 'Tutti i passi sono completati.';
    else if (rest[0].id === 'video' && !state.videos.ready) reason = 'Indica prima le cartelle dei video.';
    $('#runAll').disabled = Boolean(reason);
    $('#runAll').dataset.steps = allDone ? '' : rest.map((step) => step.id).join(',');
    $('#runAllHint').textContent = reason
      || `In fila, senza fermarsi: ${rest.map((step) => nameOf(step.id).toLowerCase()).join(', ')}.`;
  }

  function renderProject() {
    const input = $('#projectPath');
    const busy = Boolean(state.job);
    if (state.project && !dirty.has('project') && input !== document.activeElement && input.value !== state.project.path) {
      input.value = state.project.path;
    }
    const same = Boolean(state.project) && input.value.trim() === state.project.path;
    input.disabled = busy;
    $('#pickProject').disabled = busy;
    $('#useProject').disabled = busy;
    $('#useProject').classList.toggle('primary', !same);
    $('#projectNext').hidden = !same;
    $('#projectHint').textContent = state.project
      ? 'Progetto aperto: il lavoro viene salvato qui man mano. Per cambiarlo indica un\'altra cartella e premi Apri.'
      : 'Se la cartella non esiste viene creata.';

    const system = state.system;
    const notes = [];
    if (!system.checked) {
      $('#sysGpu').textContent = 'controllo in corso…';
      $('#sysFfmpeg').textContent = '–';
    } else {
      const groups = new Map();
      for (const gpu of system.gpus) {
        const key = `${gpu.name}, ${gpu.memory_gb} GB`;
        groups.set(key, (groups.get(key) || 0) + 1);
      }
      $('#sysGpu').textContent = groups.size
        ? [...groups].map(([name, n]) => (n > 1 ? `${n} × ${name}` : name)).join(' · ')
        : 'Nessuna scheda NVIDIA trovata';
      $('#sysFfmpeg').textContent = system.ffmpeg ? 'Installato' : 'Non trovato';
      if (!groups.size) {
        notes.push('Senza scheda grafica NVIDIA la ricostruzione viene fatta in versione ridotta ed è lenta: va bene solo per provare il percorso con le riprese di esempio.');
      }
      if (!system.ffmpeg) notes.push('Senza FFmpeg i video non si possono leggere: rilancia l\'installazione.');
      if (system.error) notes.push(`Dettaglio tecnico: ${system.error}`);
    }
    $('#sysDisk').textContent = state.project ? `${num(state.project.free_gb)} GB` : '–';
    if (state.project && state.project.free_gb < 20) {
      notes.push('Lo spazio libero è poco: i fotogrammi estratti da una ripresa occupano parecchi gigabyte.');
    }
    // The size of the panel follows the numbers typed in the calibration step.
    const side = Number(state.options.square_cm);
    const boardW = state.options.squares_x * side;
    const boardH = state.options.squares_y * side;
    const cm = (value) => num(value, Number.isInteger(value) ? 0 : 1);
    $('#boardSize').textContent = `Con le misure di questa pagina è largo ${cm(boardW)} cm e alto ${cm(boardH)} cm`
      + (boardW > 29.7 || boardH > 42 ? ': serve una stampa di grande formato.' : '.');
    $('#makeBoard').disabled = !state.project || busy;

    $('#reducedNote').hidden = !system.reduced;
    $('#resolutionNote').hidden = !(system.reduced && Number(state.options.downscale) < 4);
    const note = $('#sysNote');
    note.hidden = notes.length === 0;
    note.className = 'note warn';
    note.textContent = notes.join(' ');
  }

  function renderVideos() {
    const videos = state.videos;
    const locked = !state.project || Boolean(state.job);
    for (const item of $$('[data-clip]')) {
      const clip = item.dataset.clip;
      const input = $('[data-path]', item);
      const found = $('[data-found]', item);
      input.disabled = locked;
      $('[data-pick]', item).disabled = locked;
      const folder = videos ? videos.folders[clip] : '';
      const cameras = videos ? videos.cameras[clip] : [];
      if (!dirty.has(`video:${clip}`) && input !== document.activeElement && input.value !== folder) {
        input.value = folder;
        input.scrollLeft = input.scrollWidth;
      }
      found.classList.toggle('ok', cameras.length > 0);
      if (!folder) found.textContent = '';
      else if (cameras.length) found.textContent = `${num(cameras.length)} video: ${cameras.join(', ')}`;
      else found.textContent = 'In questa cartella non ci sono video.';
    }
    const box = $('#videoProblems');
    const problems = videos ? videos.problems : [];
    const key = problems.join('|');
    if (box.dataset.key !== key) {
      box.dataset.key = key;
      box.replaceChildren(el('strong', { text: 'Le cartelle non combaciano' }),
        el('ul', {}, problems.map((text) => el('li', { text }))));
    }
    box.hidden = problems.length === 0;

    // The camera kept aside for checking is chosen among those of the take.
    const select = $('#opt-holdout');
    const names = videos ? videos.cameras.take : [];
    if (select.dataset.key !== names.join('|')) {
      select.dataset.key = names.join('|');
      select.replaceChildren(el('option', { value: '', text: 'Nessuna: usa tutte le camere' }),
        ...names.map((name) => el('option', { value: name, text: name })));
    }
  }

  function fillInputs() {
    const options = state.options;
    const locked = !state.project || Boolean(state.job);
    for (const input of $$('[data-option]')) {
      const key = input.dataset.option;
      input.disabled = locked;
      if (dirty.has(key) || input === document.activeElement) continue;
      if (input.type === 'checkbox') input.checked = Boolean(options[key]);
      else {
        const value = options[key] == null ? '' : String(options[key]);
        if (input.value !== value) input.value = value;
      }
    }
    for (const name of ['sync', 'downscale']) {
      for (const radio of $$(`input[name="${name}"]`)) {
        radio.disabled = locked;
        if (!dirty.has(name)) radio.checked = radio.value === String(options[name]);
      }
    }
  }

  function renderStep(step) {
    const section = $(`.step[data-step="${step.id}"]`);
    const here = Boolean(state.job) && state.job.step === step.id;
    const actions = $('.actions', section);
    const button = $('[data-run]', section);

    let reason = '';
    if (!state.project) reason = 'Apri prima un progetto (passo 01).';
    else if (state.job) reason = 'Attendi la fine del passo in corso.';
    else if (step.blocked) {
      const first = state.steps.find((other) => other.status !== 'done');
      reason = `Prima completa il passo «${nameOf(first.id)}».`;
    } else if (step.id === 'video' && !state.videos.ready) reason = 'Indica le tre cartelle qui sopra.';
    actions.hidden = here;
    button.disabled = Boolean(reason);
    button.classList.toggle('primary', step.status !== 'done');
    button.textContent = step.status === 'done' ? 'Ripeti il passo'
      : step.status === 'error' ? 'Riprova' : button.dataset.label;
    $('.blocked-note', actions).textContent = reason;

    const box = $('.outcome', section);
    if (here) {
      renderProgress(box, state.job);
    } else {
      const key = JSON.stringify([step.status, step.problem, step.summary]);
      if (box.dataset.key !== key) {
        const justEnded = box.dataset.key === 'running';
        box.dataset.key = key;
        box.replaceChildren();
        if (step.status === 'error') box.append(problemBlock(step.problem));
        else if (step.status === 'done') box.append(RESULT[step.id](step.summary));
        if (justEnded && box.firstChild && current === step.id) {
          requestAnimationFrame(() => box.scrollIntoView({ block: 'start', behavior: 'smooth' }));
        }
      }
    }
    renderLog(step, section);
  }

  function renderProgress(box, job) {
    let block = $('.progress', box);
    if (!block) {
      box.dataset.key = 'running';
      block = el('div', { class: 'progress' },
        el('p', { class: 'what' }),
        el('div', { class: 'bar', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100' }, el('i')),
        el('div', { class: 'line' }, el('span', { class: 'detail' }),
          el('button', { class: 'btn quiet', type: 'button', text: 'Interrompi', onclick: cancel })));
      box.replaceChildren(block);
    }
    $('.what', block).textContent = job.label ? `${job.label}…` : 'Avvio…';
    const bar = $('.bar', block);
    const determinate = job.progress != null;
    bar.classList.toggle('unknown', !determinate);
    bar.setAttribute('aria-label', job.label || 'Avanzamento');
    const fill = $('i', bar);
    fill.style.width = determinate ? `${Math.round(job.progress * 100)}%` : '';
    // Length unknown: the segment advances one notch per reading, like a clock hand.
    fill.style.transform = determinate ? '' : `translateX(${(Math.floor(job.elapsed) % 8) * 100}%)`;
    if (determinate) bar.setAttribute('aria-valuenow', String(Math.round(job.progress * 100)));
    else bar.removeAttribute('aria-valuenow');
    const parts = [];
    if (job.detail) parts.push(job.detail);
    parts.push(`trascorsi ${duration(job.elapsed)}`);
    if (job.eta != null) parts.push(`ne restano circa ${duration(job.eta)}`);
    $('.detail', block).textContent = parts.join(' · ');
  }

  function renderLog(step, section) {
    const fold = $('.tech', section);
    const here = Boolean(state.job) && state.job.step === step.id;
    fold.hidden = !(here || step.status === 'done' || step.status === 'error');
    if (fold.hidden || !fold.open) return;
    const pre = $('pre', fold);
    if (here) {
      fold.dataset.key = 'live';
      pre.textContent = state.job.tail.join('\n');
      return;
    }
    const key = `${step.status}|${step.problem}|${JSON.stringify(step.summary)}`;
    if (fold.dataset.key === key) return;
    fold.dataset.key = key;
    request(`/api/log/${step.id}`).then((data) => {
      pre.textContent = data.text || 'Nessun dettaglio registrato.';
      pre.scrollTop = pre.scrollHeight;
    }).catch(() => { fold.dataset.key = ''; });
  }

  function renderExample() {
    const box = $('#exampleBox');
    const running = Boolean(state.job) && state.job.step === 'esempio';
    const empty = Boolean(state.videos) && Object.values(state.videos.folders).every((folder) => !folder);
    box.hidden = !(state.project && (running || empty));
    $('.actions', box).hidden = running;
    $('#makeExample').disabled = Boolean(state.job);
    const outcome = $('.outcome', box);
    if (running) return renderProgress(outcome, state.job);
    const key = `problem:${state.example_problem}`;
    if (outcome.dataset.key !== key) {
      outcome.dataset.key = key;
      outcome.replaceChildren(state.example_problem ? problemBlock(state.example_problem) : '');
    }
  }

  // -------------------------------------------------- outcome of each step

  function problemBlock(text) {
    return el('div', { class: 'problem', role: 'alert' },
      el('strong', { text: 'Il passo non è riuscito' }), el('p', { text }));
  }

  function resultBlock(level, title, text) {
    return el('div', { class: 'result', role: 'status' },
      el('h3', {}, el('span', { class: `verdict ${level}`, text: title })), text && el('p', { text }));
  }

  function rows(pairs) {
    return el('dl', { class: 'rows' }, pairs.filter(Boolean).map(([term, value]) =>
      el('div', {}, el('dt', { text: term }), el('dd', { text: value }))));
  }

  function table(head, body) {
    return el('table', {},
      el('thead', {}, el('tr', {}, head.map((text) => el('th', { scope: 'col', text })))),
      el('tbody', {}, body.map((cells) => el('tr', {}, cells.map((text, i) =>
        el('td', { class: i ? 'num' : '', text }))))));
  }

  // Per-camera detail is reference material: folded unless something in it needs attention.
  function detail(title, content, open) {
    return el('details', { class: 'fold inner', open }, el('summary', { text: title }), content);
  }

  function next(id) {
    const index = STEPS.findIndex((step) => step.id === id);
    const target = STEPS[index + 1];
    return el('div', { class: 'actions' }, el('button', { class: 'btn primary', type: 'button',
      text: `Continua: ${target.name.toLowerCase()}`, onclick: () => go(target.id, true) }));
  }

  const RESULT = {
    video(summary) {
      if (!summary) return el('div', {}, resultBlock('', 'Video pronti', ''), next('video'));
      const take = summary.clips.take;
      const cameras = Object.entries(take.cameras);
      const block = resultBlock('', 'Video pronti',
        `La ripresa del soggetto dura ${num(take.frames / take.fps, 1)} secondi: `
        + `${count(take.frames, 'fotogramma', 'fotogrammi')} a ${num(take.fps, Number.isInteger(take.fps) ? 0 : 2)} `
        + `al secondo da ${count(cameras.length, 'camera', 'camere')}.`);
      const synced = state.options.sync !== 'none';
      const head = ['Camera'];
      if (synced) head.push('Spostamento per allinearla', 'Scarto residuo');
      head.push('Fotogrammi persi');
      const lost = cameras.filter(([, camera]) => camera.dropped > 0).map(([name]) => name);
      if (lost.length) {
        block.append(el('p', { class: 'note warn', text: `${lost.join(', ')}: durante la registrazione sono `
          + 'andati persi dei fotogrammi. Quegli istanti vengono scartati per tutte le camere.' }));
      }
      const residual = Math.max(0, ...cameras.map(([, camera]) => Math.abs(camera.residual_ms) || 0));
      const late = synced && residual > 4;
      if (late) {
        block.append(el('p', { class: 'note warn', text: `Dopo l'allineamento resta uno scarto fino a ${num(residual, 1)} `
          + 'millisecondi tra le camere. Se il soggetto si muove in fretta, al passo 05 attiva la stima degli scarti di tempo.' }));
      }
      block.append(rows([
        ['Pannello di calibrazione', count(summary.clips.calib.frames, 'fotogramma', 'fotogrammi')],
        ['Set vuoto', count(summary.clips.empty.frames, 'fotogramma', 'fotogrammi')],
      ]));
      block.append(detail('Dettaglio per camera', table(head, cameras.map(([name, camera]) => {
        const cells = [name];
        if (synced) {
          cells.push(camera.shift ? `${signed(camera.shift)} fotogrammi` : 'nessuno',
            known(camera.residual_ms) ? `${num(Math.abs(camera.residual_ms), 1)} ms` : '–');
        }
        cells.push(camera.dropped ? num(camera.dropped) : 'nessuno');
        return cells;
      })), lost.length > 0 || late), next('video'));
      return block;
    },

    calibrazione(summary) {
      if (!summary) return el('div', {}, resultBlock('', 'Camere calibrate', ''), next('calibrazione'));
      const rms = summary.rms_px;
      const level = rms < 0.5 ? '' : rms < 1 ? 'fair' : 'poor';
      const title = { '': 'Calibrazione precisa', fair: 'Calibrazione accettabile', poor: 'Calibrazione imprecisa' }[level];
      const advice = {
        '': 'Sotto mezzo pixel è un buon valore.',
        fair: 'Funziona, ma sotto mezzo pixel il risultato è più nitido: conviene ripetere la ripresa del '
          + 'pannello muovendolo più lentamente.',
        poor: 'Sopra un pixel la scena viene sfocata: controlla le misure del pannello e ripeti la ripresa '
          + 'tenendolo fermo per un attimo in ogni posizione.',
      }[level];
      const block = resultBlock(level, title, `Errore medio di ${num(rms, 2)} pixel. ${advice}`);
      const coverage = summary.coverage;
      if (coverage) {
        const gap = `${num(coverage.largest_gap_deg)}°`;
        if (coverage.widening > 1.15) {
          block.append(el('p', { class: 'note warn', text: `Tra le direzioni di ripresa di ${coverage.between[0]} e `
            + `${coverage.between[1]} c'è un vuoto di ${gap}: visto da lì il soggetto può apparire fino al `
            + `${num((coverage.widening - 1) * 100)}% più largo del vero. Se puoi, aggiungi o sposta una camera in quel punto.` }));
        } else {
          block.append(el('p', { class: 'note', text: `Le camere circondano bene il soggetto: il vuoto più `
            + `ampio tra due direzioni di ripresa è di ${gap}.` }));
        }
      }
      block.append(detail('Errore camera per camera', table(['Camera', 'Errore'],
        Object.entries(summary.cameras).map(([name, value]) => [name, known(value) ? `${num(value, 2)} pixel` : '–'])),
        level !== ''), next('calibrazione'));
      return block;
    },

    preparazione(summary) {
      if (!summary) return el('div', {}, resultBlock('', 'Immagini pronte', ''), next('preparazione'));
      const block = resultBlock('', 'Immagini pronte',
        `${count(summary.frames, 'fotogramma', 'fotogrammi')} da ${count(summary.cameras.length, 'camera', 'camere')}, `
        + `a ${summary.size[0]} × ${summary.size[1]} pixel.`);
      block.append(el('figure', {},
        el('img', { src: summary.preview, alt: 'Anteprima delle maschere: il soggetto ritagliato dallo sfondo, camera per camera' }),
        el('figcaption', { text: 'Controlla prima di proseguire. Ogni riga è una camera: il soggetto deve restare '
          + 'intero e in piena luce, lo sfondo scurito. Se mancano parti del soggetto abbassa la soglia nelle opzioni '
          + 'avanzate, se resta chiaro dello sfondo alzala; poi ripeti il passo.' })), next('preparazione'));
      return block;
    },

    ricostruzione(summary) {
      if (!summary) return el('div', {}, resultBlock('', 'Scena ricostruita', ''), next('ricostruzione'));
      const block = resultBlock('', 'Scena ricostruita', `Ci sono voluti ${duration(summary.seconds)}.`);
      const gpus = summary.devices.filter((device) => device.startsWith('cuda'));
      const offsets = Object.entries(summary.time_offsets || {}).filter(([, value]) => Math.abs(value) >= 0.005);
      block.append(rows([
        ['Durata della scena', `${num(summary.frames / summary.fps, 1)} s (${count(summary.frames, 'fotogramma', 'fotogrammi')})`],
        ['Camere usate', num(summary.train_cameras.length)
          + (summary.holdout_cameras.length ? `, più ${summary.holdout_cameras.join(', ')} di verifica` : '')],
        ['Elementi che compongono la scena', `${num(summary.gaussians)} gaussiane`],
        ['Calcolo', gpus.length ? count(gpus.length, 'scheda grafica', 'schede grafiche')
          : 'processore, senza scheda grafica'],
        offsets.length && ['Scarti di tempo stimati',
          `${offsets.map(([name, value]) => `${name} ${signed(value, 2)}`).join(' · ')} fotogrammi`],
      ]), next('ricostruzione'));
      return block;
    },

    risultato(summary) {
      const block = resultBlock('', 'Risultato pronto', 'Nella cartella del risultato trovi player.html e orbita.mp4. '
        + 'Il player è un file unico: si copia su un altro computer e si apre con doppio clic, senza installare nulla.');
      if (!summary) return block;
      if (summary.video) {
        block.append(el('figure', {}, el('video', { src: summary.video_url, controls: true, loop: true, muted: true,
          playsinline: true, preload: 'metadata', 'aria-label': 'Video dimostrativo: la camera gira intorno alla scena' })));
      }
      block.append(el('div', { class: 'deliver' },
        summary.player && el('a', { class: 'btn primary', href: '/project/risultato/player.html', target: '_blank',
          rel: 'noopener', text: 'Apri il player' }),
        el('button', { class: 'btn', type: 'button', text: 'Apri la cartella del risultato',
          onclick: () => send('/api/open', { what: 'result' }) })));
      const wrap = el('div', {}, block);
      const quality = summary.quality;
      if (quality && known(quality.iou)) {
        const subject = known(quality.psnr_subject) ? quality.psnr_subject : quality.psnr;
        const level = quality.iou >= 0.9 && subject >= 25 ? '' : quality.iou >= 0.8 && subject >= 20 ? 'fair' : 'poor';
        const title = { '': 'Qualità buona', fair: 'Qualità discreta', poor: 'Qualità scarsa' }[level];
        const measured = resultBlock(level, title, `Misurata da ${quality.cameras.join(', ')}, che non ha partecipato `
          + 'alla ricostruzione: è il giudizio su un punto di vista che il sistema non ha mai visto.');
        measured.append(rows([
          ['Sagoma corretta', `${num(quality.iou * 100)}%`],
          known(subject) && ['Fedeltà dell\'immagine sul soggetto', `${num(subject, 1)} dB`],
          known(quality.flicker) && ['Sfarfallio tra un fotogramma e l\'altro', `${num(quality.flicker, 1)} (più basso è meglio)`],
        ]));
        if (level) {
          measured.append(el('p', { class: 'note', text: 'Di solito dipende da quanto le camere circondano il soggetto '
            + '(passo 03) o da maschere imprecise (passo 04).' }));
        }
        measured.append(el('p', { class: 'note', text: 'Per il risultato da consegnare ripeti il passo 05 senza camera '
          + 'di verifica: una camera in più migliora la scena.' }));
        wrap.append(measured);
      }
      return wrap;
    },
  };

  // ---------------------------------------------------------------- actions

  function cancel() {
    if (window.confirm('Interrompere il passo in corso? Il lavoro di questo passo andrà rifatto.')) send('/api/cancel', {});
  }

  async function openProject() {
    const path = $('#projectPath').value.trim();
    if (!path) return toast('Scrivi il percorso della cartella, oppure premi Sfoglia.');
    const data = await send('/api/project', { path });
    dirty.delete('project');
    if (data) go(defaultStep(), true);
  }

  async function pickFolder(button, title) {
    button.disabled = true;
    const data = await send('/api/pick-folder', { title });
    button.disabled = false;
    if (!data) return '';
    if (data.path == null) {
      toast('Su questo computer la finestra di scelta non si apre: scrivi o incolla il percorso della cartella.');
      return '';
    }
    return data.path;
  }

  function readOption(input) {
    if (input.type === 'checkbox') return input.checked;
    const raw = input.value.trim();
    if (input.type !== 'number') return raw;
    if (raw === '') return input.dataset.empty === 'null' ? null : undefined;
    const value = Number(raw);
    if (!Number.isFinite(value)) return undefined;
    if (input.min !== '' && value < Number(input.min)) return undefined;
    if (input.max !== '' && value > Number(input.max)) return undefined;
    return input.inputMode === 'numeric' ? Math.round(value) : value;
  }

  async function saveOption(key, value) {
    dirty.add(key);
    await send('/api/options', { [key]: value });
    dirty.delete(key);
    if (state) fillInputs();
  }

  async function saveVideo(clip, path) {
    dirty.add(`video:${clip}`);
    await send('/api/videos', { [clip]: path });
    dirty.delete(`video:${clip}`);
    if (state) renderVideos();
  }

  function listen() {
    document.addEventListener('input', (event) => {
      const input = event.target;
      if (input.matches('[data-option]')) dirty.add(input.dataset.option);
      else if (input.matches('[data-path]')) dirty.add(`video:${input.closest('[data-clip]').dataset.clip}`);
      else if (input.id === 'projectPath') { dirty.add('project'); if (state) renderProject(); }
    });

    document.addEventListener('change', (event) => {
      const input = event.target;
      if (input.matches('[data-option]')) {
        const key = input.dataset.option;
        const value = readOption(input);
        if (value === undefined) {  // empty or out of range: back to the saved value
          dirty.delete(key);
          input.value = state.options[key] == null ? '' : String(state.options[key]);
          return toast('Valore non valido: ho rimesso quello di prima.');
        }
        saveOption(key, value);
      } else if (input.matches('input[name="sync"]')) {
        saveOption('sync', input.value);
      } else if (input.matches('input[name="downscale"]')) {
        saveOption('downscale', Number(input.value));
      } else if (input.matches('[data-path]')) {
        saveVideo(input.closest('[data-clip]').dataset.clip, input.value.trim());
      }
    });

    document.addEventListener('click', async (event) => {
      const button = event.target.closest('button');
      if (!button) return;
      if (button.matches('[data-run]')) {
        follow = false;
        send('/api/run', { steps: [button.dataset.run] });
      } else if (button.matches('[data-pick]')) {
        const item = button.closest('[data-clip]');
        const path = await pickFolder(button, PICK_TITLE[item.dataset.clip]);
        if (path) {
          $('[data-path]', item).value = path;
          saveVideo(item.dataset.clip, path);
        }
      }
    });

    $('#useProject').addEventListener('click', openProject);
    $('#projectPath').addEventListener('keydown', (event) => { if (event.key === 'Enter') openProject(); });
    $('#pickProject').addEventListener('click', async (event) => {
      const path = await pickFolder(event.currentTarget, 'Cartella del progetto');
      if (path) {
        $('#projectPath').value = path;
        openProject();
      }
    });
    $('#openProject').addEventListener('click', () => send('/api/open', { what: 'project' }));
    $('#makeBoard').addEventListener('click', async (event) => {
      const button = event.currentTarget;
      const done = $('#boardDone');
      button.disabled = true;
      done.textContent = 'Creo il file…';
      const data = await send('/api/board', {});
      button.disabled = false;
      done.textContent = data && data.board
        ? `Creato ${data.board.file} nella cartella del progetto (la apri con «Apri la cartella», in alto).`
        : '';
    });
    $('#makeExample').addEventListener('click', () => { follow = false; send('/api/run', { steps: ['esempio'] }); });
    $('#runAll').addEventListener('click', (event) => {
      const steps = event.currentTarget.dataset.steps.split(',').filter(Boolean);
      if (!steps.length) return;
      follow = true;
      send('/api/run', { steps });
    });
    $('#quit').addEventListener('click', async () => {
      if (state && state.job && !window.confirm('Un passo è in corso e verrà interrotto. Uscire lo stesso?')) return;
      closed = true;
      try { await request('/api/quit', {}); } catch { /* the server may stop before answering */ }
      goodbye('4D Studio è chiuso', 'Puoi chiudere questa scheda. Il progetto resta salvato: lo ritrovi alla prossima apertura.');
    });
  }

  build();
  listen();
  poll();
})();
