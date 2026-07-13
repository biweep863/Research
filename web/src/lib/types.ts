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
  pdfHref: string | null;
};

export type Manifest = { publications: Publication[] };
