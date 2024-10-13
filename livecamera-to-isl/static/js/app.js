document.addEventListener('DOMContentLoaded', function () {
    const captureGestureBtn = document.getElementById('captureGestureBtn');
    const trainModelBtn = document.getElementById('trainModelBtn');
    const recognizeGestureBtn = document.getElementById('recognizeGestureBtn');

    // Capture Gesture
    captureGestureBtn.addEventListener('click', function () {
        const label = document.getElementById('gestureLabel').value;
        if (label) {
            fetch('/capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label })
            })
            .then(response => response.json())
            .then(data => alert(data.message))
            .catch(error => console.error('Error:', error));
        } else {
            alert('Please enter a label for the gesture.');
        }
    });

    // Train Model
    trainModelBtn.addEventListener('click', function () {
        fetch('/train', { method: 'POST' })
        .then(response => response.json())
        .then(data => alert(data.message))
        .catch(error => console.error('Error:', error));
    });

    // Recognize Gesture
    recognizeGestureBtn.addEventListener('click', function () {
        fetch('/recognize', { method: 'POST' })
        .then(response => response.json())
        .then(data => alert(data.message))
        .catch(error => console.error('Error:', error));
    });
});

// Generate random triangles
document.addEventListener("DOMContentLoaded", function() {
    const wrap = document.querySelector('.wrap');
    const totalTriangles = 1000;  // Number of triangles to be generated

    for (let i = 0; i < totalTriangles; i++) {
        const tri = document.createElement('div');
        tri.classList.add('tri');
        
        // Randomize initial positions
        tri.style.top = `${Math.random() * 100}vh`;
        tri.style.left = `${Math.random() * 100}vw`;

        // Set a random animation duration
        const duration = Math.random() * 10 + 5;  // Between 5s and 10s
        tri.style.animation = `randomMovement ${duration}s infinite`;

        wrap.appendChild(tri);
    }

    // Add event listener for mouse movement to move triangles away from the cursor
    document.addEventListener('mousemove', function(e) {
        const triangles = document.querySelectorAll('.tri');
        triangles.forEach(triangle => {
            const rect = triangle.getBoundingClientRect();
            const dx = e.clientX - (rect.left + rect.width / 2);
            const dy = e.clientY - (rect.top + rect.height / 2);
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist > 150) {  // Move away if cursor is within 150px of triangle
                const angle = Math.atan2(dy, dx);
                const offsetX = Math.cos(angle) * -50;
                const offsetY = Math.sin(angle) * -50;
                triangle.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
            }
        });
    });
});

