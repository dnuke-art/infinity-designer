import { defineConfig } from 'vite';
// base './' so the built site works from any static path.
// es2022 for top-level await in main.ts (Chrome 89+, Safari 15+, Firefox 89+)
export default defineConfig({ base: './', build: { target: 'es2022' } });
