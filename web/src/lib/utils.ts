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
  Decision: "bg-teal-50 text-teal-700 ring-teal-200",
  Energy: "bg-yellow-50 text-yellow-800 ring-yellow-200",
  Graphics: "bg-fuchsia-50 text-fuchsia-700 ring-fuchsia-200",
  Ocean: "bg-cyan-50 text-cyan-700 ring-cyan-200",
  Theory: "bg-lime-50 text-lime-800 ring-lime-200",
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
