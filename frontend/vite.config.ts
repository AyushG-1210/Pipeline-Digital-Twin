import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/Pipeline-Digital-Twin/frontend/', 
  plugins: [react()],
})
