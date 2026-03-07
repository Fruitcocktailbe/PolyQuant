/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                obsidian: '#050505',
                charcoal: '#0a0f14',
                glass: 'rgba(20, 24, 30, 0.6)',
                neon: {
                    cyan: '#00f2ff',
                    purple: '#bd00ff',
                    green: '#00ff94',
                    red: '#ff0055',
                }
            },
            fontFamily: {
                mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'Courier New', 'monospace'],
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
            boxShadow: {
                'neon-cyan': '0 0 10px rgba(0, 242, 255, 0.3), 0 0 20px rgba(0, 242, 255, 0.1)',
                'neon-red': '0 0 10px rgba(255, 0, 85, 0.3), 0 0 20px rgba(255, 0, 85, 0.1)',
                'glass-inset': 'inset 0 0 20px rgba(255, 255, 255, 0.02)',
            },
            animation: {
                'pulse-fast': 'pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'glitch': 'glitch 1s linear infinite',
            }
        },
    },
    plugins: [],
}
