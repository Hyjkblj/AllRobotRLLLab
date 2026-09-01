import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  define: {
    __PLATFORM_API_TARGET__: JSON.stringify(process.env.VITE_PLATFORM_API_TARGET || "http://127.0.0.1:8010"),
    __MUJOCO_API_TARGET__: JSON.stringify(process.env.VITE_MUJOCO_API_TARGET || "http://127.0.0.1:8787")
  },
  server: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
    proxy: {
      "/api/v1": { target: process.env.VITE_PLATFORM_API_TARGET || "http://127.0.0.1:8010", changeOrigin: true },
      "/api/mujoco": { target: process.env.VITE_MUJOCO_API_TARGET || "http://127.0.0.1:8787", changeOrigin: true },
      "/uploads": { target: process.env.VITE_PLATFORM_API_TARGET || "http://127.0.0.1:8010", changeOrigin: true },
      "/objects": { target: process.env.VITE_PLATFORM_API_TARGET || "http://127.0.0.1:8010", changeOrigin: true }
    }
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true
  }
});
