import { createMcpHandler } from "mcp-handler";
import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";

// Minimal example tools; extend with your own app logic
const handler = createMcpHandler(
  (server) => {

    // Return the main documentation content in markdown format
    server.tool(
      "docs_get_all",
      "Get the main StruktX documentation content. Returns clean markdown documentation.",
      {},
      async () => {
        try {
          const docsPath = path.join(process.cwd(), "app", "docs", "MDDOCS.md");
          const data = await fs.readFile(docsPath, "utf8");
          return { content: [{ type: "text", text: data }] };
        } catch (err: any) {
          return { content: [{ type: "text", text: `Error: ${err?.message || String(err)}` }] };
        }
      }
    );

    // Simple full-text search across docs files
    server.tool(
      "docs_search",
      "Search documentation for a query string and return top matches.",
      { query: z.string().min(2) },
      async ({ query }) => {
        const root = process.cwd();
        const docsDir = path.join(root, "docs");
        const includeExtraFiles = [path.join(root, "README.md"), path.join(root, "app", "docs", "page.tsx")];

        async function* walk(dir: string): AsyncGenerator<string> {
          const entries = await fs.readdir(dir, { withFileTypes: true });
          for (const e of entries) {
            const fp = path.join(dir, e.name);
            if (e.isDirectory()) {
              yield* walk(fp);
            } else if (/\.(md|mdx|json)$/i.test(e.name)) {
              yield fp;
            }
          }
        }

        const files: string[] = [];
        try {
          for await (const f of walk(docsDir)) files.push(f);
        } catch {}
        files.push(...includeExtraFiles);

        const q = query.toLowerCase();
        const results: string[] = [];
        for (const f of files) {
          try {
            const txt = await fs.readFile(f, "utf8");
            const idx = txt.toLowerCase().indexOf(q);
            if (idx !== -1) {
              const start = Math.max(0, idx - 200);
              const end = Math.min(txt.length, idx + 200);
              const snippet = txt.slice(start, end).replace(/\s+/g, " ");
              results.push(`- ${path.relative(root, f)}: … ${snippet} …`);
            }
          } catch {}
          if (results.length >= 8) break;
        }
        if (results.length === 0) return { content: [{ type: "text", text: "No matches found." }] };
        return { content: [{ type: "text", text: results.join("\n") }] };
      }
    );
  },
  {},
  {
    basePath: "/api",
    maxDuration: 60,
    verboseLogs: true,
  }
);

export { handler as GET, handler as POST };


