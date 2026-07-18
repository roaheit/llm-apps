import type { ToolDefinition } from "../types";

export const fileRead: ToolDefinition = {
  name: "file_read",
  description:
    "Read the contents of a URL or file path. For URLs (http/https), fetches the page and returns the text body. " +
    "For local paths, reads using the File API if available.",
  parameters: [
    {
      name: "path",
      type: "string",
      description: "URL (https://...) or file path to read",
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

    try {
      if (path.startsWith("http://") || path.startsWith("https://")) {
        const response = await fetch(path);
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
      }
      return `Local file reading is not supported in the browser. Use a URL instead, or provide a custom file_read tool with Node.js fs access.`;
    } catch (e) {
      return `Read failed: ${(e as Error).message}`;
    }
  },
};
