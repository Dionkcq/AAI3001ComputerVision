// Object Detection Interface Handler
document.addEventListener('DOMContentLoaded', () => {
  // Mode switching
  const modeButtons = document.querySelectorAll('.mode-btn');
  const sections = {
    'mode-webcam': document.getElementById('webcam-section'),
    'mode-video': document.getElementById('video-section'),
    'mode-image': document.getElementById('image-section')
  };

  modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      // Update button styles
      modeButtons.forEach(b => {
        b.classList.remove('active', 'bg-white', 'text-zinc-900');
        b.classList.add('bg-zinc-800', 'text-white');
      });
      btn.classList.add('active', 'bg-white', 'text-zinc-900');
      btn.classList.remove('bg-zinc-800', 'text-white');

      // Show/hide sections
      Object.values(sections).forEach(s => s?.classList.add('hidden'));
      sections[btn.id]?.classList.remove('hidden');

      // Stop webcam if switching away
      if (btn.id !== 'mode-webcam') {
        stopWebcam();
      }
    });
  });

  // ===== WEBCAM MODE =====
  const startWebcamBtn = document.getElementById('start-webcam');
  const stopWebcamBtn = document.getElementById('stop-webcam');
  const webcamStream = document.getElementById('webcam-stream');
  const webcamStatus = document.getElementById('webcam-status');
  let webcamInterval = null;

  startWebcamBtn?.addEventListener('click', startWebcam);
  stopWebcamBtn?.addEventListener('click', stopWebcam);

  function startWebcam() {
    webcamStatus.textContent = 'Starting webcam...';
    webcamStream.src = '/webcam_feed';
    webcamStream.classList.remove('hidden');
    startWebcamBtn.classList.add('hidden');
    stopWebcamBtn.classList.remove('hidden');
    webcamStatus.textContent = 'Webcam active - detecting exercises...';
    
    // Handle errors
    webcamStream.onerror = () => {
      webcamStatus.textContent = 'Error: Could not access webcam';
      stopWebcam();
    };
  }

  function stopWebcam() {
    if (webcamStream) {
      webcamStream.src = '';
      webcamStream.classList.add('hidden');
    }
    if (webcamInterval) {
      clearInterval(webcamInterval);
      webcamInterval = null;
    }
    startWebcamBtn?.classList.remove('hidden');
    stopWebcamBtn?.classList.add('hidden');
    webcamStatus.textContent = '';
  }

  // ===== VIDEO UPLOAD MODE =====
  const videoForm = document.getElementById('video-form');
  const videoInput = document.getElementById('video-input');
  const videoStream = document.getElementById('video-stream');
  const videoStatus = document.getElementById('video-status');

  videoForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = videoInput.files?.[0];
    if (!file) return;

    videoStatus.textContent = 'Uploading video...';
    
    try {
      const formData = new FormData();
      formData.append('video', file);

      const res = await fetch('/upload_video', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      if (!res.ok || !data.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      videoStatus.textContent = 'Processing video...';
      videoStream.src = `/video_feed/${data.video_id}`;
      videoStream.classList.remove('hidden');
      videoStatus.textContent = 'Video processing complete!';

    } catch (err) {
      console.error(err);
      videoStatus.textContent = `Error: ${err.message}`;
    }
  });

  // ===== IMAGE UPLOAD MODE =====
  const imageForm = document.getElementById('image-form');
  const imageInput = document.getElementById('image-input');
  const previewBox = document.getElementById('preview');
  const previewImg = document.getElementById('preview-img');
  const imageStatus = document.getElementById('image-status');
  const imageResult = document.getElementById('image-result');

  // Preview handler
  imageInput?.addEventListener('change', () => {
    const file = imageInput.files?.[0];
    if (!file) return;
    
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewBox.classList.remove('hidden');
    imageStatus.textContent = '';
  });

  // Submit handler
  imageForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = imageInput.files?.[0];
    if (!file) return;

    imageStatus.textContent = 'Analyzing image...';
    imageResult.innerHTML = '<div class="text-zinc-400 text-sm">Processing...</div>';

    try {
      const formData = new FormData();
      formData.append('image', file);

      const res = await fetch('/analyze_image', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      if (!res.ok || !data.ok) {
        throw new Error(data.error || 'Analysis failed');
      }

      // Display results
      const detections = data.detections || [];
      const detectionCount = detections.length;
      
      let resultsHTML = `
        <div class="space-y-4">
          <div class="flex justify-center">
            <img src="${data.annotated_image}" 
                 alt="Detected exercises" 
                 class="rounded-lg border border-white/10 max-w-full">
          </div>
          
          <div class="text-lg font-semibold">
            Found ${detectionCount} exercise${detectionCount !== 1 ? 's' : ''}
          </div>
      `;

      if (detections.length > 0) {
        resultsHTML += '<div class="space-y-3">';
        
        detections.forEach((det, idx) => {
          const conf = (det.confidence * 100).toFixed(1);
          resultsHTML += `
            <div class="bg-zinc-800/50 rounded-lg p-4 border border-white/5">
              <div class="flex justify-between items-center">
                <span class="font-semibold text-lg capitalize">${det.label}</span>
                <span class="text-zinc-400">${conf}% confident</span>
              </div>
            </div>
          `;
        });
        
        resultsHTML += '</div>';

        // Add tips
        if (data.tips && data.tips.length > 0) {
          resultsHTML += '<div class="mt-4 space-y-2">';
          resultsHTML += '<div class="font-semibold">Form Tips:</div>';
          data.tips.forEach(tip => {
            if (tip) {
              resultsHTML += `<p class="text-zinc-300 text-sm">• ${tip}</p>`;
            }
          });
          resultsHTML += '</div>';
        }
      }

      resultsHTML += '</div>';
      imageResult.innerHTML = resultsHTML;
      imageStatus.textContent = '';

    } catch (err) {
      console.error(err);
      imageStatus.textContent = `Error: ${err.message}`;
      imageResult.innerHTML = '<div class="text-red-400 text-sm">Analysis failed. Please try again.</div>';
    }
  });
});
