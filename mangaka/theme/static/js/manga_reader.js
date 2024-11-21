document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const dropZoneContainer = document.getElementById("drop-zone-container");
  const fileInput = document.getElementById("file-input");
  const uploadButton = document.getElementById("upload-button");
  const previewer = document.getElementById("previewer");
  const previewImage = document.getElementById("preview-image");
  const prevButton = document.getElementById("prev-button");
  const nextButton = document.getElementById("next-button");
  const imageSidebar = document.getElementById("image-sidebar");
  const thumbnailsContainer = document.getElementById("thumbnails");
  const importImagesButton = document.getElementById("import-images-button");

  let images = []; // Stocke les images sous forme de base64
  let currentIndex = 0;

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

  const createThumbnail = (src, index) => {
    const thumbnail = document.createElement("img");
    thumbnail.src = src;
    thumbnail.classList.add(
      "w-full",
      "h-auto",
      "rounded",
      "cursor-pointer",
      "hover:opacity-75",
      "transition"
    );
    thumbnail.addEventListener("click", () => {
      currentIndex = index;
      updatePreview();
    });
    return thumbnail;
  };

  const handleFiles = (files, replace = false) => {
    if (replace) {
      images = []; // Réinitialise les images
      currentIndex = 0;
      thumbnailsContainer.innerHTML = ""; // Efface les miniatures existantes
    }

    const fileArray = Array.from(files).sort((a, b) =>
      a.name.localeCompare(b.name)
    ); // Trie les fichiers par ordre alphabétique

    fileArray.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        images.push(e.target.result);
        thumbnailsContainer.appendChild(
          createThumbnail(e.target.result, images.length - 1)
        );
        if (images.length === fileArray.length) {
          updatePreview();

          // Masque la zone de drop et affiche la barre latérale
          dropZoneContainer.classList.add("hidden");
          imageSidebar.classList.remove("hidden");
          imageSidebar.classList.add("flex");
        }
      };
      reader.readAsDataURL(file);
    });
  };

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("bg-gray-700");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("bg-gray-700");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("bg-gray-700");
    handleFiles(e.dataTransfer.files, true); // Remplace les images actuelles
  });

  // Corrige le comportement du bouton "Upload"
  uploadButton.addEventListener("click", () => {
    fileInput.click(); // Simule un clic sur l'élément file-input
  });

  fileInput.addEventListener("change", (e) => {
    handleFiles(e.target.files, true); // Remplace les images actuelles
  });

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

  importImagesButton.addEventListener("click", () => {
    fileInput.click();
  });
});
