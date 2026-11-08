import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The middle layer sends no CORS headers, on purpose: that is half of what stops
// an arbitrary web page from driving it (see guard in internal/api). So the page
// is never cross-origin to it. Every API call goes to /api on this server, and
// Vite forwards it. The Host it forwards is loopback, which the guard accepts.
const middleLayer = process.env.FORGE_MIDDLE_LAYER ?? "http://127.0.0.1:8080";

const proxy = {
  "/api": {
    target: middleLayer,
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  // Loopback only, like the middle layer: this page can start sandboxed code.
  server: { host: "127.0.0.1", port: 5173, strictPort: true, proxy },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true, proxy },
});
