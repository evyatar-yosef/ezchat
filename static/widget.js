
(function(){
  const TENANT = document.currentScript.getAttribute('data-tenant') || 'default';
  const API_BASE = document.currentScript.getAttribute('data-api') || '';

  const style = document.createElement('style');
  style.textContent = `
    .ez-chat-btn{position:fixed;right:18px;bottom:18px;border:none;border-radius:999px;padding:12px 16px;box-shadow:0 6px 20px rgba(0,0,0,.15);background:#111;color:#fff;cursor:pointer;z-index:999999}
    .ez-chat-box{position:fixed;right:18px;bottom:78px;width:320px;max-height:60vh;background:#fff;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.15);overflow:hidden;display:none;flex-direction:column;z-index:999999}
    .ez-chat-header{padding:10px 12px;border-bottom:1px solid #eee;font-weight:600;background:#f8fafc}
    .ez-chat-body{padding:10px 12px;overflow:auto;display:flex;flex-direction:column;gap:8px}
    .ez-msg{font-size:14px;line-height:1.35;padding:8px 10px;border-radius:12px;max-width:80%}
    .ez-me{align-self:flex-end;background:#111;color:#fff}
    .ez-bot{align-self:flex-start;background:#f1f5f9}
    .ez-input{display:flex;gap:6px;padding:10px;border-top:1px solid #eee}
    .ez-input input{flex:1;border:1px solid #e5e7eb;border-radius:10px;padding:8px 10px}
    .ez-input button{border:none;background:#111;color:#fff;border-radius:10px;padding:8px 12px;cursor:pointer}
  `;
  document.head.appendChild(style);

  const btn = document.createElement('button');
  btn.className = 'ez-chat-btn';
  btn.textContent = 'צ׳אט';

  const box = document.createElement('div');
  box.className = 'ez-chat-box';
  box.innerHTML = `
    <div class="ez-chat-header">שירות לקוחות</div>
    <div class="ez-chat-body" id="ez-body"></div>
    <div class="ez-input">
      <input id="ez-inp" placeholder="שאל/י כל דבר..." />
      <button id="ez-send">שלח</button>
    </div>`;

  document.body.appendChild(btn);
  document.body.appendChild(box);

  btn.onclick = () => {
    box.style.display = (box.style.display==='flex'?'none':'flex');
    if(box.style.display==='flex'){ box.style.display='flex'; box.querySelector('#ez-inp').focus(); }
  };

  const body = box.querySelector('#ez-body');
  const input = box.querySelector('#ez-inp');
  const send = box.querySelector('#ez-send');

  function push(role, text){
    const div = document.createElement('div');
    div.className = 'ez-msg ' + (role==='user'?'ez-me':'ez-bot');
    div.textContent = text;
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
  }

  async function ask(){
    const q = (input.value||'').trim();
    if(!q) return;
    push('user', q);
    input.value = '';
    push('assistant', 'חושב...');

    try{
      const res = await fetch(API_BASE + '/api/chat', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({tenant_id: TENANT, message: q})
      });
      const data = await res.json();
      body.lastChild.textContent = data.answer || 'תקלה בתשובה';
    }catch(err){
      body.lastChild.textContent = 'שגיאה בחיבור לשרת';
    }
  }

  input.addEventListener('keydown', e=>{ if(e.key==='Enter') ask(); });
  send.addEventListener('click', ask);
})();
