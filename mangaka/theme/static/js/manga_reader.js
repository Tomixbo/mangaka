document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const uploadButton = document.getElementById("upload-button");
  const previewer = document.getElementById("previewer");
  const previewImage = document.getElementById("preview-image");
  const prevButton = document.getElementById("prev-button");
  const nextButton = document.getElementById("next-button");

  let images = []; // Store images as base64
  let currentIndex = 0;

  // Show preview of the current image
  const updatePreview = () => {
    if (images.length > 0) {
      previewImage.src = images[currentIndex];
      previewer.classList.remove("hidden");
      previewer.classList.add("flex");
    } else {
      previewer.classList.remove("flex");
      previewer.classList.add("hidden");
    }
  };

  // Handle file uploads
  const handleFiles = (files) => {
    images = [];
    currentIndex = 0;

    const fileArray = Array.from(files).sort((a, b) =>
      a.name.localeCompare(b.name)
    ); // Sort files alphabetically

    fileArray.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        images.push(e.target.result);
        if (images.length === fileArray.length) {
          updatePreview(); // Update preview once all images are loaded
        }
      };
      reader.readAsDataURL(file);
    });
  };

  // Drag and Drop functionality
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("bg-gray-200");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("bg-gray-200");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("bg-gray-200");
    handleFiles(e.dataTransfer.files);
  });

  // Click to upload functionality
  uploadButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => handleFiles(e.target.files));

  // Navigation buttons
  prevButton.addEventListener("click", () => {
    if (currentIndex > 0) {
      currentIndex -= 1;
      updatePreview();
    }
  });

  nextButton.addEventListener("click", () => {
    if (currentIndex < images.length - 1) {
      currentIndex += 1;
      updatePreview();
    }
  });
});
