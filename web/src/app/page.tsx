"use client";

import { useMemo, useState } from "react";
import manifest from "@/data/publications.json";
import type { Manifest, Publication } from "@/lib/types";
import { PaperCard } from "@/components/PaperCard";
import { PaperModal } from "@/components/PaperModal";
import { FilterBar, type Filters } from "@/components/FilterBar";

const AUTHOR_NAME = "Luis Benvenuto";
const AUTHOR_INITIALS = "LB";

const ALL = "__all__";
const publications = (manifest as Manifest).publications as Publication[];

function unique<T>(arr: T[]) {
  return Array.from(new Set(arr));
}

export default function HomePage() {
  const [filters, setFilters] = useState<Filters>({
    q: "",
    year: ALL,
    topic: ALL,
    type: ALL,
    sort: "recent",
  });
  const [openPub, setOpenPub] = useState<Publication | null>(null);

  const years = useMemo(
    () =>
      unique(publications.map((p) => String(p.year))).sort(
        (a, b) => Number(b) - Number(a),
      ),
    [],
  );
  const topics = useMemo(
    () => unique(publications.map((p) => p.topic)).sort(),
    [],
  );
  const types = useMemo(
    () => unique(publications.map((p) => p.type)).sort(),
    [],
  );

  const filtered = useMemo(() => {
    const q = filters.q.trim().toLowerCase();
    const list = publications.filter((p) => {
      if (filters.year !== ALL && String(p.year) !== filters.year) return false;
      if (filters.topic !== ALL && p.topic !== filters.topic) return false;
      if (filters.type !== ALL && p.type !== filters.type) return false;
      if (!q) return true;
      const haystack = [
        p.title,
        p.tldr,
        p.authors.join(" "),
        p.topic,
        p.venue ?? "",
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(q);
    });
    if (filters.sort === "recent")
      list.sort((a, b) => b.year - a.year || a.title.localeCompare(b.title));
    if (filters.sort === "oldest")
      list.sort((a, b) => a.year - b.year || a.title.localeCompare(b.title));
    if (filters.sort === "az")
      list.sort((a, b) => a.title.localeCompare(b.title));
    return list;
  }, [filters]);

  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="flex items-center gap-3 border-b border-slate-200 pb-6">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-900 text-sm font-semibold text-white">
          {AUTHOR_INITIALS}
        </div>
        <span className="text-sm font-medium text-slate-700">
          {AUTHOR_NAME} — Papers
        </span>
      </header>

      <section className="mt-8">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          Scientific publications
        </h1>
        <p className="mt-2 text-sm text-slate-500">
          {publications.length} articles, proceedings and preprints.
        </p>

        <div className="mt-6">
          <FilterBar
            filters={filters}
            setFilters={setFilters}
            years={years}
            topics={topics}
            types={types}
          />
        </div>

        {filtered.length === 0 ? (
          <div className="mt-10 rounded-xl border border-dashed border-slate-300 p-10 text-center text-sm text-slate-500">
            No publications match the current filters.
          </div>
        ) : (
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {filtered.map((pub) => (
              <PaperCard key={pub.slug} pub={pub} onOpen={setOpenPub} />
            ))}
          </div>
        )}
      </section>

      <PaperModal pub={openPub} onClose={() => setOpenPub(null)} />
    </main>
  );
}
