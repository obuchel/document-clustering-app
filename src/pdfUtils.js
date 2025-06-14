

// Import PDF.js
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf';

// Set worker source
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/5.3.31/pdf.worker.min.js';

// Initialize PDF.js
function initPdfJs() {
  return pdfjsLib;
}

// Parse PDF file
async function parsePDF(file) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    let fullText = '';
    
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items
        .map(item => 'str' in item ? item.str : '')
        .join(' ');
      fullText += pageText + ' ';
    }
    
    console.log('PDF Text extracted:', fullText.substring(0, 500) + '...'); // Debug log
    return { title: file.name, content: fullText };
  } catch (error) {
    console.error('PDF parsing error:', error);
    throw new Error('Failed to parse PDF: ' + error.message);
  }
}

// Export functions using ES Module syntax consistently
export {
  initPdfJs,
  parsePDF
};

// Removed the global window assignments for initPdfJs and parsePDF.
// They are now imported directly as ES modules in App.js,
// so global exposure is unnecessary and can cause conflicts.
