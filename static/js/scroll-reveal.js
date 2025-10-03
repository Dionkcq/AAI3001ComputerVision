// Fade-in on scroll (IntersectionObserver)
(() => {
  const els = document.querySelectorAll('.reveal');
  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.remove('opacity-0', 'translate-y-4');
        entry.target.classList.add('transition', 'duration-700', 'ease-out');
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.18 });

  els.forEach(el => io.observe(el));
})();