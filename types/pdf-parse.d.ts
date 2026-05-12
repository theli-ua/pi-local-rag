// pdf-parse ships no types. The inner path bypasses the index.js debug-mode
// side effect (which tries to read a missing test PDF when imported as main).
declare module "pdf-parse/lib/pdf-parse.js" {
  const pdf: (data: Buffer) => Promise<{ text: string }>;
  export default pdf;
}

// Internal pdfjs build that pdf-parse loads on demand. Imported only so we can
// drop verbosity to errors-only — see silencePdfjsWarnings() in index.ts.
declare module "pdf-parse/lib/pdf.js/v1.10.100/build/pdf.js" {
  const m: { PDFJS?: { verbosity?: number } };
  export default m;
}
