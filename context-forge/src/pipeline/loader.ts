import type { DocumentSource, RawDocument } from "../types";

let docCounter = 0;

function makeId() {
  return `doc_${++docCounter}_${Date.now()}`;
}

async function loadFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
    reader.readAsText(file);
  });
}

async function loadUrl(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch URL ${url}: ${res.status}`);
  const contentType = res.headers.get("content-type") ?? "";
  if (contentType.includes("text/html")) {
    const html = await res.text();
    // Strip HTML tags for plain text extraction
    return html
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }
  return res.text();
}

export async function loadSource(source: DocumentSource): Promise<RawDocument> {
  const id = makeId();

  if (source.type === "text") {
    const text = source.content as string;
    return {
      id,
      sourceType: "text",
      sourceName: (source.metadata?.name as string) ?? `Text document ${id}`,
      rawText: text,
      metadata: source.metadata,
    };
  }

  if (source.type === "file") {
    const file = source.content as File;
    const rawText = await loadFile(file);
    return {
      id,
      sourceType: "file",
      sourceName: file.name,
      rawText,
      metadata: { ...source.metadata, fileName: file.name, fileSize: file.size },
    };
  }

  if (source.type === "url") {
    const url = source.content as string;
    const rawText = await loadUrl(url);
    return {
      id,
      sourceType: "url",
      sourceName: url,
      rawText,
      metadata: { ...source.metadata, url },
    };
  }

  throw new Error(`Unknown source type: "${(source as DocumentSource).type}"`);
}
