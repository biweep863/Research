"use client";

import { Input } from "./ui/Input";
import { Select } from "./ui/Select";

export type Filters = {
  q: string;
  year: string;
  topic: string;
  type: string;
  sort: "recent" | "oldest" | "az";
};

type Props = {
  filters: Filters;
  setFilters: (f: Filters) => void;
  years: string[];
  topics: string[];
  types: string[];
};

const opts = (values: string[], allLabel: string) => [
  { value: "__all__", label: allLabel },
  ...values.map((v) => ({ value: v, label: v })),
];

export function FilterBar({ filters, setFilters, years, topics, types }: Props) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-[1fr_repeat(4,minmax(0,10rem))]">
      <Input
        placeholder="Buscar por título, autor o palabra clave"
        value={filters.q}
        onChange={(e) => setFilters({ ...filters, q: e.target.value })}
      />
      <Select
        label="Año"
        value={filters.year}
        onChange={(v) => setFilters({ ...filters, year: v })}
        options={opts(years, "todos")}
      />
      <Select
        label="Tema"
        value={filters.topic}
        onChange={(v) => setFilters({ ...filters, topic: v })}
        options={opts(topics, "todos")}
      />
      <Select
        label="Tipo"
        value={filters.type}
        onChange={(v) => setFilters({ ...filters, type: v })}
        options={opts(types, "todos")}
      />
      <Select
        value={filters.sort}
        onChange={(v) => setFilters({ ...filters, sort: v as Filters["sort"] })}
        options={[
          { value: "recent", label: "Más reciente" },
          { value: "oldest", label: "Más antiguo" },
          { value: "az", label: "A–Z" },
        ]}
      />
    </div>
  );
}
