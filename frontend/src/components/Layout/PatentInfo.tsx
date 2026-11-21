export const PATENT_APPLICATION_NO = '202641135823';
export const PATENT_PDF_URL = 'https://ayushgouda.me/assets/Amorflux_Patent-CwcmqMdQ.pdf';

/**
 * Compact "Patent Published" pill for the header.
 * Links to the published patent PDF in a new tab.
 */
export function PatentBadge() {
    return (
        <a
            href={PATENT_PDF_URL}
            target="_blank"
            rel="noopener noreferrer"
            title={`Patent application ${PATENT_APPLICATION_NO} — published. Open the PDF.`}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-line bg-surface hover:bg-surface-sunken text-xs font-semibold text-ink transition-all shadow-sm group"
        >
            <svg className="w-4 h-4 text-primary group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <span>Patent Published</span>
        </a>
    );
}

/**
 * Slim footer strip: demo disclaimer, patent number, and PDF link.
 */
export function AppFooter() {
    return (
        <footer className="bg-surface border-t border-line px-6 py-1.5 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-[11px] text-ink-muted">
            <span>
                &copy; {new Date().getFullYear()} Amorflux &middot; Demo environment
            </span>
            <span className="flex items-center gap-2">
                <span>Patent application No. <span className="font-mono text-ink-soft">{PATENT_APPLICATION_NO}</span> (published)</span>
                <span aria-hidden="true">&middot;</span>
                <a
                    href={PATENT_PDF_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline font-medium"
                >
                    View patent (PDF)
                </a>
            </span>
        </footer>
    );
}
