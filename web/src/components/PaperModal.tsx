"use client";

import { useEffect, useRef } from "react";
import type { Publication } from "@/lib/types";
import { cn, topicColor, withBasePath } from "@/lib/utils";

type Props = {
  pub: Publication | null;
  onClose: () => void;
};

function formatDate(iso: string) {
  // Render as e.g. "Jun 16, 2026" in a locale-agnostic way.
  const [y, m, d] = iso.split("-").map(Number);
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[m - 1]} ${d}, ${y}`;
}

export function PaperModal({ pub, onClose }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!pub) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
      previouslyFocused?.focus?.();
    };
  }, [pub, onClose]);

  if (!pub) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4 backdrop-blur-sm"
      onClick={onClose}
      aria-hidden={false}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="paper-modal-title"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-2xl overflow-hidden rounded-2xl bg-white shadow-xl outline-none"
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-100 p-6">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
                topicColor(pub.topic),
              )}
            >
              {pub.topic}
            </span>
            <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-200">
              {pub.type}
            </span>
            <span className="text-sm text-slate-500">{pub.year}</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          >
            <svg viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5" aria-hidden>
              <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
            </svg>
          </button>
        </div>

        <div className="max-h-[70vh] overflow-y-auto p-6">
          <h2 id="paper-modal-title" className="text-xl font-semibold leading-snug text-slate-900">
            {pub.title}
          </h2>
          <p className="mt-2 text-sm text-slate-500">{pub.authors.join(", ")}</p>

          <p className="mt-5 text-sm leading-relaxed text-slate-700">{pub.tldr}</p>

          <dl className="mt-6 grid grid-cols-1 gap-4 border-t border-slate-100 pt-5 sm:grid-cols-2">
            {pub.venue && (
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-400">Venue</dt>
                <dd className="mt-1 text-sm text-slate-700">{pub.venue}</dd>
              </div>
            )}
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-400">Type</dt>
              <dd className="mt-1 text-sm text-slate-700">{pub.type}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-400">Created</dt>
              <dd className="mt-1 text-sm text-slate-700">{formatDate(pub.created)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-400">Last updated</dt>
              <dd className="mt-1 text-sm text-slate-700">{formatDate(pub.updated)}</dd>
            </div>
          </dl>
        </div>

        <div className="flex items-center justify-end gap-3 border-t border-slate-100 bg-slate-50 px-6 py-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-3 py-2 text-sm font-medium text-slate-600 hover:text-slate-900"
          >
            Close
          </button>
          {pub.pdfHref ? (
            <a
              href={withBasePath(pub.pdfHref)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800"
            >
              Open PDF
              <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4" aria-hidden>
                <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
              </svg>
            </a>
          ) : (
            <span className="text-xs text-slate-400">PDF pending</span>
          )}
        </div>
      </div>
    </div>
  );
}
