document.addEventListener("DOMContentLoaded", () => {
  // Retrieve the panels data from the script tag
  const panelsDataElement = document.getElementById("panels-data");
  let panels = [];

  if (panelsDataElement) {
    try {
      panels = JSON.parse(panelsDataElement.textContent);
      console.log("Panels data loaded:", panels);
    } catch (error) {
      console.error("Error parsing panels data:", error);
      panels = [];
    }
  } else {
    console.error("Panels data element not found.");
  }

  // Initialize variables
  let currentIndex = 0;
  let playInterval = null;

  // Get references to the DOM elements
  const previewImage = document.getElementById("preview-image");
  const prevButton = document.getElementById("prev-button");
  const nextButton = document.getElementById("next-button");
  const firstButton = document.getElementById("first-button");
  const lastButton = document.getElementById("last-button");
  const playButton = document.getElementById("play-button");
  const slider = document.getElementById("slider");
  const playOverlay = document.getElementById("play-overlay");

  // Update slider range and value
  slider.max = panels.length - 1;

  // Function to update the preview image
  function updatePreview() {
    if (panels.length > 0 && previewImage) {
      const panelBase64 = panels[currentIndex];
      previewImage.src = `data:image/jpeg;base64,${panelBase64}`;
      slider.value = currentIndex; // Sync slider with current index
    } else {
      console.warn("No panels to display or preview image element not found.");
    }
  }

  // Function to toggle play/pause
  function togglePlayPause() {
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
      playButton.innerHTML = '<i class="fas fa-play"></i>'; // Change icon to play
      playButton.classList.replace("bg-green-600", "bg-gray-600");
      playOverlay.classList.add("hidden"); // Hide play overlay
    } else {
      playInterval = setInterval(() => {
        if (currentIndex < panels.length - 1) {
          currentIndex++;
          updatePreview();
        } else {
          togglePlayPause(); // Automatically pause at the last panel
        }
      }, 3000); // Change every 3 seconds
      playButton.innerHTML = '<i class="fas fa-pause"></i>'; // Change icon to pause
      playButton.classList.replace("bg-gray-600", "bg-green-600");
      playOverlay.classList.remove("hidden"); // Show play overlay
    }
  }

  // Event listeners for navigation buttons
  if (firstButton) {
    firstButton.addEventListener("click", () => {
      currentIndex = 0;
      updatePreview();
    });
  }

  if (prevButton) {
    prevButton.addEventListener("click", () => {
      if (currentIndex > 0) {
        currentIndex--;
        updatePreview();
      }
    });
  }

  if (nextButton) {
    nextButton.addEventListener("click", () => {
      if (currentIndex < panels.length - 1) {
        currentIndex++;
        updatePreview();
      }
    });
  }

  if (lastButton) {
    lastButton.addEventListener("click", () => {
      currentIndex = panels.length - 1;
      updatePreview();
    });
  }

  if (playButton) {
    playButton.addEventListener("click", togglePlayPause);
  }

  // Event listener for the slider
  slider.addEventListener("input", (e) => {
    currentIndex = parseInt(e.target.value);
    updatePreview();
  });

  // Initial call to display the first panel
  updatePreview();
});
