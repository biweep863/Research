import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const TOPIC_COLORS: Record<string, string> = {
  Geometry: "bg-indigo-50 text-indigo-700 ring-indigo-200",
  Interpretability: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  Efficiency: "bg-amber-50 text-amber-800 ring-amber-200",
  Architecture: "bg-sky-50 text-sky-700 ring-sky-200",
  Attention: "bg-rose-50 text-rose-700 ring-rose-200",
  Topology: "bg-violet-50 text-violet-700 ring-violet-200",
};

export function topicColor(topic: string) {
  return TOPIC_COLORS[topic] ?? "bg-slate-100 text-slate-700 ring-slate-200";
}

export function withBasePath(href: string) {
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  if (!href) return href;
  if (href.startsWith("http") || href.startsWith("/")) return href;
  return `${base}/${href}`;
}
