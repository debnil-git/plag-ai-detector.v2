````markdown name=README.md
# Debnil's Open Source Plagiarism & AI Detector v10.4

A desktop application to detect both plagiarism and AI-generated content in text, with detailed visualizations and support for TXT, DOCX, and PDF files. This project is built using Python, Tkinter for the GUI, and state-of-the-art AI/NLP models for analysis.

---

## Features

- **Full Analysis:** Detects both plagiarism (via web search similarity) and AI-generated content (via GPT-2 perplexity).
- **AI-Only Mode:** Fast detection of AI-generated text without plagiarism checking.
- **Multiple Graph Types:** Visualize results with pie charts, trends, radar charts, and more.
- **File Support:** Load and analyze `.txt`, `.docx`, and `.pdf` files.
- **Batch Analysis:** Analyzes text sentence by sentence for comprehensive results.
- **Modern GUI:** Built with Tkinter, with live status, progress bar, and easy controls.

---

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/debnil-git/plag-ai-detector.v2.git
cd plag-ai-detector.v2
```

### 2. Install Dependencies

It's recommended to use a Python 3.8+ virtual environment.

```sh
pip install -r requirements.txt
```

**Typical requirements:**
- `torch`
- `transformers`
- `sentence-transformers`
- `aiohttp`
- `beautifulsoup4`
- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`
- `docx2txt`
- `PyMuPDF` (for `fitz`)
- `fpdf`
- `Pillow`
- `tkinter` (usually included with Python)

> _If you encounter issues with PyMuPDF, use:_  
> `pip install pymupdf`

---

## Usage

Run the main application:

```sh
python main.py
```

### Features on Startup

- **Enter text directly** or **load a file** (TXT, DOCX, PDF).
- Click **"Full Analysis"** for plagiarism + AI detection.
- Click **"AI Only"** for fast AI-generated content detection.
- View detailed results and graphical analysis.
- **Save** your results or **clear** the workspace.

---

## How It Works

- **Plagiarism Check:**  
  Each sentence is searched online (DuckDuckGo) to find similar content. Cosine similarity is computed between the sentence and retrieved web content.

- **AI Detection:**  
  Uses GPT-2 language model to compute perplexity; lower perplexity suggests AI-generated content.

- **Visualization:**  
  Multiple graphs (pie, trend, radar, stacked bar, etc.) help interpret the results.

---

## Notes & Recommendations

- **First run requires internet** to download NLP models (hosted by Hugging Face).
- For **offline use**, pre-download and cache models (see Hugging Face docs).
- **Do not overuse** the plagiarism checking function—it sends web queries for every sentence.
- If running as a standalone `.exe` (via PyInstaller), large models may not be bundled; see documentation for packaging tips.

---

## Screenshots

> _Add screenshots of the GUI and example output here._

---

## License

Open source for all. Use wisely and responsibly.

---

## Credits

- [Hugging Face Transformers](https://huggingface.co/)
- [Sentence Transformers](https://www.sbert.net/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)

---

## Contributing

Pull requests and suggestions are welcome! Please open an issue for feedback or questions.

---

## Disclaimer

This tool is for educational and research purposes only. Results depend on model accuracy and availability of web search.

````