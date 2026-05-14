export const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
export const VECTOR_DIM = 384;

export const TEXT_EXTS = new Set([
  ".md", ".txt", ".ts", ".js", ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".cs",
  ".css", ".html", ".json", ".yaml", ".yml", ".toml", ".xml", ".csv", ".sh",
  ".sql", ".graphql", ".proto", ".env", ".gitignore", ".dockerfile",
]);

export const BINARY_DOC_EXTS = new Set([".pdf", ".docx"]);

export const TEXT_MAX_BYTES = 500_000;
export const BINARY_DOC_MAX_BYTES = 10_000_000;

export const SKIP_DIRS = new Set([
  "node_modules", ".git", ".next", "dist", "build", "__pycache__", ".venv", "venv", ".cache",
]);
