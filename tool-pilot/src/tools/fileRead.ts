import type { ToolDefinition } from "../types";

/** Block loopback / private / link-local hosts to reduce SSRF risk. */
function isBlockedHost(hostname: string): boolean {
  const h = hostname.toLowerCase().replace(/^\[|\]$/g, ""); // strip IPv6 brackets
  if (h === "localhost" || h.endsWith(".localhost") || h === "" || h === "0.0.0.0" || h === "::1" || h === "::") {
    return true;
  }
  const m = h.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (m) {
    const a = Number(m[1]);
    const b = Number(m[2]);
    if (a === 0 || a === 127 || a === 10) return true;
    if (a === 192 && b === 168) return true;
    if (a === 169 && b === 254) return true; // link-local, incl. cloud metadata 169.254.169.254
    if (a === 172 && b >= 16 && b <= 31) return true;
  }
  if (h.startsWith("fc") || h.startsWith("fd") || h.startsWith("fe80")) return true; // IPv6 ULA / link-local
  return false;
}

export const fileRead: ToolDefinition = {
  name: "file_read",
  description:
    "Fetch the text content of an http(s) URL and return it (HTML tags stripped). " +
    "Runs in the caller's environment and is subject to CORS in the browser; requests to " +
    "loopback/private-network addresses are blocked to reduce SSRF risk.",
  parameters: [
    {
      name: "path",
      type: "string",
      description: "URL (https://...) to read",
      required: true,
    },
    {
      name: "maxLength",
      type: "number",
      description: "Maximum characters to return (default: 4000). Content is truncated if longer.",
      required: false,
    },
  ],
  execute: async (args) => {
    const path = args.path as string;
    const maxLength = (args.maxLength as number) ?? 4000;
    if (!path) return "Error: path is required";

    if (!(path.startsWith("http://") || path.startsWith("https://"))) {
      return "Local file reading is not supported in the browser. Use an http(s) URL, or provide a custom file_read tool with Node.js fs access.";
    }

    let url: URL;
    try {
      url = new URL(path);
    } catch {
      return `Error: invalid URL "${path}"`;
    }
    if (isBlockedHost(url.hostname)) {
      return (
        `Blocked: refusing to fetch loopback/private-network host "${url.hostname}" (SSRF guard). ` +
        `Provide a custom file_read tool if you must reach internal hosts.`
      );
    }

    try {
      const response = await fetch(url.toString());
      if (!response.ok) return `Fetch failed: HTTP ${response.status}`;
      let text = await response.text();
      // Strip HTML tags for readability
      text = text.replace(/<script[\s\S]*?<\/script>/gi, "");
      text = text.replace(/<style[\s\S]*?<\/style>/gi, "");
      text = text.replace(/<[^>]+>/g, " ");
      text = text.replace(/\s+/g, " ").trim();
      if (text.length > maxLength) {
        text = text.slice(0, maxLength) + "\n...(truncated)";
      }
      return text || "(empty response)";
    } catch (e) {
      return `Read failed: ${(e as Error).message}`;
    }
  },
};
