document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const resultEl = document.getElementById('result');
    if (!form) return;

    form.addEventListener('submit', async function (event) {
        event.preventDefault(); // Prevent page reload
        const formData = new FormData(form);

        try {
            const res = await fetch('/analyze', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.ok) {
                // Update HTML with prediction (adjust selector as needed)
                resultEl.textContent = `Prediction: ${data.prediction.label} (${(data.prediction.confidence * 100).toFixed(1)}%)`;
            } else {
                resultEl.textContent = 'Error: ' + (data.error || 'Analysis failed.');
            }
        } catch (err) {
            resultEl.textContent = 'Network Error';
        }
    });
});