export type Publication = {
  slug: string;
  title: string;
  authors: string[];
  year: number;
  topic: string;
  type: string;
  venue?: string;
  tldr: string;
  pdf: string;
  created: string;    // ISO YYYY-MM-DD
  updated: string;    // ISO YYYY-MM-DD
  pdfHref: string | null;
};

export type Manifest = { publications: Publication[] };
