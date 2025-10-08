document.addEventListener('DOMContentLoaded', () => {
  const form       = document.getElementById('upload-form');
  const fileInput  = document.getElementById('file-input');
  const previewBox = document.getElementById('preview');
  const previewImg = document.getElementById('preview-img');
  const statusEl   = document.getElementById('upload-status');
  const resultEl   = document.getElementById('result');
  const submitBtn  = form?.querySelector('button[type="submit"]');

  if (!form) return;

  // Local preview
  let lastPreviewSrc = null;
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    if (!f) return;
    lastPreviewSrc = URL.createObjectURL(f);
    previewImg.src = lastPreviewSrc;
    previewBox.classList.remove('hidden');
    statusEl.textContent = '';
  });

  let busy = false;
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (busy) return;
    busy = true;
    if (submitBtn) submitBtn.disabled = true;
    statusEl.textContent = 'Analyzing…';
    resultEl.innerHTML = '';

    try {
      const fd = new FormData(form);               // contains "image"
      const url = form.action || '/analyze';       // action set in HTML
      const res = await fetch(url, { method: 'POST', body: fd });

      let data;
      try { data = await res.json(); }
      catch { throw new Error(`Bad JSON (HTTP ${res.status})`); }

      if (!res.ok || data?.ok === false) {
        throw new Error(data?.error || `HTTP ${res.status}`);
      }

      const pred = data?.prediction?.label ?? data?.label ?? '—';
      const conf = data?.prediction?.confidence ?? data?.confidence;
      const confTxt = (conf != null) ? ` (${(Number(conf) * 100).toFixed(1)}%)` : '';

      // Prefer server-provided image; fall back to local preview
      const imgSrc = data.image_url
        ? data.image_url
        : (data.image_b64 ? `data:image/png;base64,${data.image_b64}` : lastPreviewSrc);

      resultEl.innerHTML = `
        <div class="grid md:grid-cols-2 gap-4 items-start">
          ${imgSrc ? `<img src="${imgSrc}" alt="Analyzed image"
               class="rounded-lg border border-white/10 max-h-72 object-contain bg-black/20">` : ''}
          <div>
            <div class="text-xl font-semibold">Prediction: ${pred}${confTxt}</div>
            ${data.form?.note ? `<p class="mt-2 text-zinc-400">${data.form.note}</p>` : ''}
            ${data.tip ? `<p class="mt-2 text-zinc-300">${data.tip}</p>` : ''}
          </div>
        </div>
      `;
      statusEl.textContent = '';
    } catch (err) {
      console.error(err);
      statusEl.textContent = err.message || 'Something went wrong. Please try again.';
    } finally {
      busy = false;
      if (submitBtn) submitBtn.disabled = false;
    }
  });
});
