/** @type {import('tailwindcss').Config} */
export default {
    darkMode: 'class',
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Themable tokens — backed by CSS variables in index.css so a
                // single `.dark` class on <html> re-themes everything that
                // uses these names. rgb(var(...) / <alpha-value>) keeps
                // opacity modifiers (e.g. bg-ink/10) working.
                'canvas': 'rgb(var(--color-canvas) / <alpha-value>)',
                'surface': {
                    DEFAULT: 'rgb(var(--color-surface) / <alpha-value>)',
                    sunken: 'rgb(var(--color-surface-sunken) / <alpha-value>)',
                },
                'line': {
                    DEFAULT: 'rgb(var(--color-line) / <alpha-value>)',
                    strong: 'rgb(var(--color-line-strong) / <alpha-value>)',
                },
                'ink': {
                    DEFAULT: 'rgb(var(--color-ink) / <alpha-value>)',       // headings, high-emphasis text
                    soft: 'rgb(var(--color-ink-soft) / <alpha-value>)',     // body text
                    muted: 'rgb(var(--color-ink-muted) / <alpha-value>)',   // secondary / placeholder text
                },
                'primary': {
                    DEFAULT: 'rgb(var(--color-primary) / <alpha-value>)',
                    hover: 'rgb(var(--color-primary-hover) / <alpha-value>)',
                    soft: 'rgb(var(--color-primary-soft) / <alpha-value>)',
                },
                'secondary': {
                    DEFAULT: 'rgb(var(--color-secondary) / <alpha-value>)',
                    hover: 'rgb(var(--color-secondary-hover) / <alpha-value>)',
                    soft: 'rgb(var(--color-secondary-soft) / <alpha-value>)',
                },
                // Accent — used sparingly: outlines, small icons, focus/active states only
                'accent': {
                    DEFAULT: 'rgb(var(--color-accent) / <alpha-value>)',
                    soft: 'rgb(var(--color-accent-soft) / <alpha-value>)',
                },
                // Status colors for integrity / RUL benchmarking
                'healthy': {
                    DEFAULT: '#4C9A78',
                    dark: '#3C7C61',
                    light: '#E4F2EA',
                },
                'warning': {
                    DEFAULT: '#D98E2B',
                    dark: '#B5721B',
                    light: '#FBEEDA',
                },
                'critical': {
                    DEFAULT: '#D6473C',
                    dark: '#B33529',
                    light: '#FBE4E1',
                },
                // Kept for the 3D viewport chrome, which intentionally stays dark for contrast
                'industrial': {
                    950: '#050505',
                    900: '#0f0f10',
                    850: '#171718',
                    800: '#202022',
                    750: '#2a2a2d',
                    700: '#38383c',
                    600: '#4d4d52',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            },
        },
    },
    plugins: [],
}
