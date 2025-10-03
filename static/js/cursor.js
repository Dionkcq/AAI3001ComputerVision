(() => {
  // Keep native cursor on touch devices
  if ('ontouchstart' in window) return;

  // Hide the old dot and ring (blurry circle)
  const dot = document.getElementById('cursor-dot');
  const ring = document.getElementById('cursor-ring');
  if (dot) dot.style.display = 'none';
  if (ring) ring.style.display = 'none';

  // Create a simple white triangle cursor
  const cursor = document.createElement('div');
  cursor.style.position = 'fixed';
  cursor.style.left = '0';
  cursor.style.top = '0';
  cursor.style.width = '0';
  cursor.style.height = '0';
  cursor.style.pointerEvents = 'none';
  cursor.style.zIndex = '50';
  cursor.style.transition = 'transform 0.08s ease-out';
  cursor.innerHTML = `
    <svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <!-- Classic arrow-style cursor pointing top-left -->
      <polygon points="1,1 1,21 6,16 9,22 12,21 9,15 19,15" fill="white" shape-rendering="crispEdges"/>
    </svg>
  `;
  document.body.appendChild(cursor);

  // Smooth follow
  let mx = window.innerWidth / 2;
  let my = window.innerHeight / 2;
  let ax = mx, ay = my;
  const lerp = (a, b, t) => a + (b - a) * t;

  function raf() {
    ax = lerp(ax, mx, 0.25);
    ay = lerp(ay, my, 0.25);
    cursor.style.transform = `translate(${ax}px, ${ay}px)`;
    requestAnimationFrame(raf);
  }
  requestAnimationFrame(raf);

  window.addEventListener('mousemove', (e) => {
    mx = e.clientX;
    my = e.clientY;
  });
})();