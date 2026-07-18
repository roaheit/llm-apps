import type { ToolDefinition } from "../types";

export const webSearch: ToolDefinition = {
  name: "web_search",
  description: "Search the web for current information. Returns a summary of top results.",
  parameters: [
    {
      name: "query",
      type: "string",
      description: "The search query",
      required: true,
    },
    {
      name: "maxResults",
      type: "number",
      description: "Maximum number of results to return (default: 5)",
      required: false,
    },
  ],
  execute: async (args) => {
    const query = args.query as string;
    if (!query) return "Error: query is required";

    // Default implementation: calls a public search-summary API.
    // Override by providing your own ToolDefinition with a custom execute().
    try {
      const response = await fetch(
        `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1`
      );
      const data = await response.json();

      const results: string[] = [];
      if (data.AbstractText) {
        results.push(`Summary: ${data.AbstractText}`);
      }
      if (data.RelatedTopics) {
        const topics = data.RelatedTopics.slice(0, (args.maxResults as number) ?? 5);
        for (const topic of topics) {
          if (topic.Text) results.push(`- ${topic.Text}`);
        }
      }
      return results.length > 0
        ? results.join("\n")
        : `No results found for "${query}". Try a different query.`;
    } catch (e) {
      return `Search failed: ${(e as Error).message}`;
    }
  },
};
