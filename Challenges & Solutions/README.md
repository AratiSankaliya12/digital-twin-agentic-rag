
## 🚧 Challenges & Solutions

### 1. The "Ghost Data" Problem
**Issue:** Old files remained in the Vector DB even after being deleted from the source folder, leading to outdated answers or conflicting information.

**Solution:** Implemented a robust "Nuclear Clean-up" protocol using `shutil` that wipes and rebuilds the `chroma_db` directory on initialization, ensuring 100% data synchronicity.

### 2. The "Crowding Out" Effect
**Issue:** When retrieving too many document chunks (high `k` value), irrelevant text would "crowd out" the specific answer in the context window, causing the model to miss key details ("Lost in the Middle" phenomenon).

**Solution:** Optimized the retrieval parameters (tuned `k` and `score_threshold`) and refined chunk sizes to ensure only high-quality, relevant context reaches the LLM.

### 3. Hallucinations on Code
**Issue:** The model initially ignored local `.py` files and gave generic coding advice (e.g., standard library usage) instead of explaining the specific custom logic in the project.

**Solution:** A strict System Prompt was engineered: *"PRIORITY: Check Context First. Do not answer from general knowledge if the answer is in the retrieved documents."*

### 4. Complex File Parsing (Image-based PDFs)
**Issue:** Standard PDF loaders returned empty strings for scanned documents or files with complex layouts, creating knowledge gaps.

**Solution:** Integrated OCR-capable preprocessing and built a custom file router that detects file types and switches between standard text extraction and OCR-based loading when necessary.
