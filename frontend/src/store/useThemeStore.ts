import { create } from 'zustand';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'pipeline-twin-theme';

function getInitialTheme(): Theme {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored === 'light' || stored === 'dark') return stored;
    } catch {
        // localStorage unavailable — fall through to default
    }
    return 'light';
}

function applyTheme(theme: Theme) {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    try {
        localStorage.setItem(STORAGE_KEY, theme);
    } catch {
        // ignore write failures (private mode, etc.)
    }
}

interface ThemeStore {
    theme: Theme;
    toggleTheme: () => void;
    setTheme: (theme: Theme) => void;
}

const initial = getInitialTheme();
applyTheme(initial);

export const useThemeStore = create<ThemeStore>((set, get) => ({
    theme: initial,
    toggleTheme: () => {
        const next: Theme = get().theme === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        set({ theme: next });
    },
    setTheme: (theme) => {
        applyTheme(theme);
        set({ theme });
    },
}));
