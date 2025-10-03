// Animated dumbbell particles on a Canvas background
(() => {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d', { alpha: true });

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(innerWidth * dpr);
    canvas.height = Math.floor(innerHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener('resize', resize);
  resize();

  // Create particles shaped like tiny dumbbells
  const NUM = Math.min(70, Math.floor((innerWidth * innerHeight) / 22000));
  const particles = Array.from({ length: NUM }, () => spawn());

  function spawn() {
    const s = 0.5 + Math.random() * 1.2; // scale
    return {
      x: Math.random() * innerWidth,
      y: Math.random() * innerHeight,
      vx: -0.4 + Math.random() * 0.8,
      vy: -0.4 + Math.random() * 0.8,
      rot: Math.random() * Math.PI * 2,
      vr: (-0.004 + Math.random() * 0.008),
      s,
      alpha: 0.15 + Math.random() * 0.35
    };
  }

  function drawDumbbell(x, y, scale, rot, alpha) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rot);
    ctx.scale(scale, scale);
    ctx.globalAlpha = alpha;

    // bar
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.beginPath();
    ctx.moveTo(-16, 0);
    ctx.lineTo(16, 0);
    ctx.stroke();

    // plates
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.beginPath(); ctx.arc(-18, 0, 4, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(18, 0, 4, 0, Math.PI*2); ctx.fill();

    ctx.restore();
  }

  function tick() {
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    for (const p of particles) {
      p.x += p.vx; p.y += p.vy; p.rot += p.vr;

      // wrap-around
      if (p.x < -30) p.x = innerWidth + 30;
      if (p.x > innerWidth + 30) p.x = -30;
      if (p.y < -30) p.y = innerHeight + 30;
      if (p.y > innerHeight + 30) p.y = -30;

      drawDumbbell(p.x, p.y, p.s, p.rot, p.alpha);
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();