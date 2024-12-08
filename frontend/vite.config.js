import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  css: {
    preprocessorOptions: {
      tailwind: {
        config: "./tailwind.config.js",
      },
    },
  },
});