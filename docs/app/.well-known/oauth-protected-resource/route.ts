import {
  protectedResourceHandler,
  metadataCorsOptionsRequestHandler,
} from "mcp-handler";

const handler = protectedResourceHandler({
  authServerUrls: [process.env.MCP_AUTH_ISSUER || "https://auth-server.com"],
});

export { handler as GET, metadataCorsOptionsRequestHandler as OPTIONS };


