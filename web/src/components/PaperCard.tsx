import type { Publication } from "@/lib/types";
import { cn, topicColor, withBasePath } from "@/lib/utils";

export function PaperCard({ pub }: { pub: Publication }) {
  return (
    <article className="flex flex-col rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-3 flex items-start justify-between gap-3">
        <span
          className={cn(
            "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
            topicColor(pub.topic),
          )}
        >
          {pub.topic}
        </span>
        <span className="text-sm text-slate-500">{pub.year}</span>
      </div>

      <h3 className="text-base font-semibold leading-snug text-slate-900">{pub.title}</h3>

      <p className="mt-1 text-sm text-slate-500">{pub.authors.join(", ")}</p>

      <p className="mt-3 text-sm leading-relaxed text-slate-700">{pub.tldr}</p>

      <div className="mt-4 flex items-end justify-between border-t border-slate-100 pt-3">
        <span className="text-xs text-slate-500">
          {pub.venue ? `${pub.venue} · ${pub.type}` : pub.type}
        </span>
        {pub.pdfHref ? (
          <a
            href={withBasePath(pub.pdfHref)}
            target="_blank"
            rel="noreferrer"
            className="text-sm font-medium text-blue-700 hover:text-blue-900"
          >
            PDF ↓
          </a>
        ) : (
          <span className="text-xs text-slate-400">PDF pendiente</span>
        )}
      </div>
    </article>
  );
}
