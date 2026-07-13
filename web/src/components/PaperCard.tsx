"use client";

import type { Publication } from "@/lib/types";
import { cn, topicColor } from "@/lib/utils";

type Props = {
  pub: Publication;
  onOpen: (pub: Publication) => void;
};

export function PaperCard({ pub, onOpen }: Props) {
  return (
    <button
      type="button"
      onClick={() => onOpen(pub)}
      className={cn(
        "group flex w-full flex-col rounded-2xl border border-slate-200 bg-white p-5 text-left shadow-sm",
        "transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400",
      )}
      aria-label={`Open details for ${pub.title}`}
    >
      <div className="flex items-center justify-between gap-3">
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

      <h3 className="mt-3 text-base font-semibold leading-snug text-slate-900 line-clamp-2 group-hover:text-slate-950">
        {pub.title}
      </h3>

      <span className="mt-auto pt-4 text-xs font-medium text-slate-400 group-hover:text-slate-600">
        View details →
      </span>
    </button>
  );
}
