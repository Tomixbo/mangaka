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

  // Get references to the DOM elements
  const previewImage = document.getElementById("preview-image");
  const prevButton = document.getElementById("prev-button");
  const nextButton = document.getElementById("next-button");

  // Function to update the preview image
  function updatePreview() {
    if (panels.length > 0 && previewImage) {
      const panelBase64 = panels[currentIndex];
      previewImage.src = `data:image/jpeg;base64,${panelBase64}`;
    } else {
      console.warn("No panels to display or preview image element not found.");
    }
  }

  // Event listeners for navigation buttons
  if (prevButton) {
    prevButton.addEventListener("click", (e) => {
      e.preventDefault();
      if (currentIndex > 0) {
        currentIndex--;
        updatePreview();
      }
    });
  }

  if (nextButton) {
    nextButton.addEventListener("click", (e) => {
      e.preventDefault();
      if (currentIndex < panels.length - 1) {
        currentIndex++;
        updatePreview();
      }
    });
  }

  // Initial call to display the first panel
  updatePreview();
});
