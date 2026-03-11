// static/app.js (revisi kecil: hapus duplikat admin, isi #admin-wa-link)
// Semua logika chat asli dipertahankan.

const api = {
  categories: '/api/categories',
  suggest:    '/api/suggest-questions',
  chat:       '/api/chat',
  health:     '/health',
  admin:      '/api/admin'
};

let currentCategory = null;
let sessionId = null;
let ADMIN_WA = '';

/* Utilities */
function wait(ms){ return new Promise(r => setTimeout(r, ms)); }
function timestampNow(){
  const d = new Date();
  return d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}
function logErrorToUI(msg){
  const m = document.getElementById('messages');
  if (m){
    const el = document.createElement('div');
    el.className = 'msg bot';
    el.style.border = '2px solid #ffcccc';
    el.style.background = '#fff0f0';
    el.textContent = msg;
    m.appendChild(el);
    m.scrollTop = m.scrollHeight;
  }
}

/* Chat UI helpers */
function makeMsgContainer(role, text){
  const cont = document.getElementById('messages');
  const wrapper = document.createElement('div');
  wrapper.className = 'msg ' + (role === 'user' ? 'user' : 'bot');

  const p = document.createElement('div');
  p.className = 'msg-text';
  p.textContent = text;
  wrapper.appendChild(p);

  const ts = document.createElement('div');
  ts.className = 'msg-ts';
  ts.textContent = timestampNow();
  ts.style.fontSize = '11px';
  ts.style.color = '#6b7280';
  ts.style.marginTop = '6px';
  wrapper.appendChild(ts);

  cont.appendChild(wrapper);
  cont.scrollTop = cont.scrollHeight;
  return wrapper;
}

function addMessage(role, text){
  return makeMsgContainer(role, text);
}

function showWelcome(){
  addMessage('bot', 'Halo! Saya asisten virtual BKKBN. Silakan pilih kategori di samping atau klik salah satu pertanyaan yang disarankan untuk memulai.');
}

function showTypingIndicator(){
  const cont = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'msg bot typing-indicator';
  el.innerHTML = '<span class="dot">•</span> <span class="dot">•</span> <span class="dot">•</span> Sedang mengetik...';
  cont.appendChild(el);
  cont.scrollTop = cont.scrollHeight;
  return () => { try { el.remove(); } catch(_){} };
}

async function animatedBotReply(text){
  const cont = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'msg bot';
  const p = document.createElement('div'); p.className = 'msg-text';
  const ts = document.createElement('div'); ts.className = 'msg-ts'; ts.textContent = timestampNow();
  ts.style.fontSize='11px'; ts.style.color='#6b7280'; ts.style.marginTop='6px';
  el.appendChild(p);
  el.appendChild(ts);
  cont.appendChild(el);
  cont.scrollTop = cont.scrollHeight;

  const words = (text + "").split(/\s+/);
  p.textContent = '';
  for (let i = 0; i < words.length; i++){
    p.textContent += (i === 0 ? "" : " ") + words[i];
    cont.scrollTop = cont.scrollHeight;
    await wait(35 + Math.random()*60);
  }
  return el;
}

/* Categories & Suggestions */
async function loadHealth(){
  try {
    const r = await fetch(api.health);
    const j = await r.json();
    const el = document.getElementById('health');
    if (el) el.textContent = j.status === 'ok' ? 'online' : 'offline';
  } catch (e){
    const el = document.getElementById('health');
    if (el) el.textContent = 'offline';
  }
}

// loadAdmin: ambil nomor/wa_link dari server dan isi elemen #admin-wa-link (single button)
async function loadAdmin(){
  try {
    const r = await fetch(api.admin);
    const j = await r.json();
    const wa = (j && j.wa) ? j.wa : '';
    const wa_link = (j && j.wa_link) ? j.wa_link : (`https://wa.me/${encodeURIComponent((wa||'').replace(/\D/g,''))}`);

    ADMIN_WA = wa || '';

    // update the single admin button in DOM (if present)
    const a = document.getElementById('admin-wa-link');
    if (a){
      a.href = wa_link;
      a.textContent = 'Hubungi Admin';
    }
  } catch (e){
    // fallback: set link to default phone
    ADMIN_WA = '6287819985271';
    const fallback = `https://wa.me/${encodeURIComponent(ADMIN_WA)}`;
    const a = document.getElementById('admin-wa-link');
    if (a){
      a.href = fallback;
      a.textContent = 'Hubungi Admin';
    }
  }
}

async function loadCategories(){
  try {
    const r = await fetch(api.categories);
    const j = await r.json();
    renderCategories(j.categories || []);
  } catch (e){
    console.error('loadCategories error', e);
    renderCategories([]);
  }
}

// renderCategories: TIDAK membuat tombol admin (hanya membuat pill kategori)
function renderCategories(cats){
  const cont = document.getElementById('category-pills');
  if (!cont) return;
  cont.innerHTML = '';

  function makePill(name){
    const btn = document.createElement('button');
    btn.className = 'pill';
    btn.type = 'button';
    btn.textContent = name;
    btn.dataset.cat = name;
    return btn;
  }

  cont.appendChild(makePill('Semua Kategori'));

  if (!cats || cats.length === 0){
    cont.appendChild(makePill('Umum'));
  } else {
    cats.forEach(c => {
      const nm = (typeof c.name === 'string') ? c.name.trim() : String(c.name);
      cont.appendChild(makePill(nm));
    });
  }

  // delegate click
  cont.removeEventListener('click', pillClickHandler);
  cont.addEventListener('click', pillClickHandler);
}

function pillClickHandler(ev){
  const btn = ev.target.closest('.pill');
  if (!btn) return;
  const name = btn.dataset.cat || btn.textContent;

  // set active
  document.querySelectorAll('.pill').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');

  if (name === 'Semua Kategori') currentCategory = null;
  else if (name === 'Umum') currentCategory = 'Umum';
  else currentCategory = name;

  showSuggestionsForCategory(currentCategory || 'Umum').catch(e => console.error(e));
}

async function showSuggestionsForCategory(categoryName){
  try {
    const url = new URL(api.suggest, window.location.origin);
    if (categoryName) url.searchParams.set('category', categoryName);
    url.searchParams.set('limit', 5);
    const r = await fetch(url.toString());
    const j = await r.json();
    const list = j.questions || [];

    // remove previous suggestion bubbles for categories
    document.querySelectorAll('.suggestion-bubble[data-category]').forEach(n => {
      if (n.dataset.category !== (categoryName || 'Umum')) n.remove();
    });

    if (!list || !list.length) return;
    const cont = document.getElementById('messages');
    if (!cont) return;

    const bubble = document.createElement('div');
    bubble.className = 'suggestion-bubble bot';
    bubble.dataset.category = categoryName || 'Umum';

    const header = document.createElement('div');
    header.className = 'suggest-header';
    header.textContent = `Beberapa pertanyaan di kategori ${categoryName}`;
    bubble.appendChild(header);

    const listDiv = document.createElement('div');
    listDiv.className = 'suggest-list';

    for (const q of list){
      const btn = document.createElement('button');
      btn.className = 'suggest-btn';
      btn.type = 'button';
      btn.textContent = (typeof q === 'string') ? q : (q.text || q.question || '');
      btn.addEventListener('click', ()=>{
        addMessage('user', btn.textContent);
        listDiv.querySelectorAll('.suggest-btn').forEach(b => b.classList.remove('active-sugg'));
        btn.classList.add('active-sugg');
        sendMessage(btn.textContent);
      });
      listDiv.appendChild(btn);
    }

    bubble.appendChild(listDiv);
    cont.appendChild(bubble);
    cont.scrollTop = cont.scrollHeight;
  } catch (e){
    console.warn('showSuggestionsForCategory error', e);
  }
}

/* Composer & Sending */
function setupComposer(){
  const form = document.getElementById('composer');
  const input = document.getElementById('inputText');

  if (!form || !input) {
    console.warn('composer or inputText element not found');
    return;
  }

  form.addEventListener('submit', async (ev)=>{
    ev.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    addMessage('user', text);
    input.value = '';
    await sendMessage(text);
  });

  input.addEventListener('keydown', (e)=>{
    if (e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      form.dispatchEvent(new Event('submit', {cancelable:true}));
    }
  });
}

async function sendMessage(msg){
  const clearTyping = showTypingIndicator();

  try {
    const body = { message: msg, session_id: sessionId };
    if (currentCategory) body.category = currentCategory;
    const res = await fetch(api.chat, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });

    if (!res.ok){
      clearTyping();
      addMessage('bot', 'Kesalahan server. Kode: ' + res.status);
      return;
    }

    const j = await res.json();
    if (j.session_id) sessionId = j.session_id;

    // keep typing visible for at least 1s
    await wait(1000);

    // remove typing indicator
    clearTyping();

    // If server indicates out-of-context, show apology + WA link
    if (j.ooc || j.wa_link){
      // show provided message(s) first (if any)
      if (j.messages && j.messages.length){
        for (const m of j.messages){
          const text = (m.text || m.answer || '') + "";
          await animatedBotReply(text);
        }
      } else if (j.messages === undefined && j.answer){
        // older variant: server embedded wa link in answer text
        await animatedBotReply(j.answer + "");
      } else {
        // default apology message
        await animatedBotReply('Maaf, pertanyaan ini belum tersedia di FAQ.');
      }

      // render clickable WA link (use server-provided or fallback)
      const wa = j.wa_link || (`https://wa.me/${encodeURIComponent((ADMIN_WA||'').replace(/\D/g,'')) || '6287819985271'}`);
      const el = document.createElement('div');
      el.className = 'msg bot';
      const a = document.createElement('a');
      a.href = wa;
      a.target = '_blank';
      a.rel = 'noopener';
      a.textContent = 'Hubungi Admin via WhatsApp';
      a.style.fontWeight = '700';
      el.appendChild(a);
      const ts = document.createElement('div'); ts.className='msg-ts'; ts.textContent = timestampNow(); ts.style.fontSize='11px'; ts.style.color='#6b7280'; ts.style.marginTop='6px';
      el.appendChild(ts);
      document.getElementById('messages').appendChild(el);
      document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
      return;
    }

    // handle messages array
    if (j.messages && j.messages.length){
      for (const m of j.messages){
        const text = (m.text || m.answer || '') + "";
        await animatedBotReply(text);
      }
      return;
    }

    // server may include direct answer
    if (j.answer){
      // if answer contains wa.me link, make clickable
      const answer = j.answer + "";
      if (answer.includes('https://wa.me/')){
        const waMatch = answer.match(/https:\/\/wa\.me\/[^\s)]+/);
        if (waMatch){
          const before = answer.substring(0, waMatch.index);
          const link = waMatch[0];
          if (before && before.trim()) await animatedBotReply(before.trim());
          const el = document.createElement('div');
          el.className = 'msg bot';
          const a = document.createElement('a');
          a.href = link;
          a.target = '_blank';
          a.rel = 'noopener';
          a.textContent = 'Hubungi Admin via WhatsApp';
          a.style.fontWeight = '700';
          el.appendChild(a);
          const ts = document.createElement('div'); ts.className='msg-ts'; ts.textContent = timestampNow(); ts.style.fontSize='11px'; ts.style.color='#6b7280'; ts.style.marginTop='6px';
          el.appendChild(ts);
          document.getElementById('messages').appendChild(el);
          document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
          return;
        }
      }
      await animatedBotReply(answer);
      return;
    }

    // fallback: top retrieved
    if (j.retrieved && j.retrieved.length){
      const top = j.retrieved[0];
      const text = (top.answer || top.answer_text || top.text || 'Maaf, saya belum dapat menjawab.') + "";
      await animatedBotReply(text);
      return;
    }

    // default fallback
    await animatedBotReply('Maaf, saya belum dapat menjawab pertanyaan tersebut.');
  } catch (e){
    console.error('sendMessage error', e);
    try { /* attempt to clear typing */ } catch(_) {}
    addMessage('bot', 'Kesalahan jaringan atau server: ' + (e.message || e));
  }
}

/* Init */
async function init(){
  try {
    const required = ['category-pills','messages','composer','inputText','health'];
    for (const id of required){
      if (!document.getElementById(id)){
        throw new Error(`Element with id="${id}" not found in HTML. Check template.`);
      }
    }

    await loadAdmin();      // fetch admin WA number from server and update single button
    await loadHealth();
    await loadCategories();
    setupComposer();
    showWelcome();
    await showSuggestionsForCategory('Umum');
    console.log('init done');
  } catch (e){
    console.error('init() failed:', e);
    logErrorToUI('Init error: ' + e.message);
  }
}

window.addEventListener('load', init);
