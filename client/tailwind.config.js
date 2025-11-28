/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: "#0E1117",
        card: "#262730",
        accent: "#00f2ea",
        secondary: "#ff0055",
      }
    },
  },
  plugins: [],
}