#!/usr/bin/env node
// Scans ../projects/*/meta.yaml, validates, copies PDFs into public/papers/,
// and writes src/data/publications.json for the app to consume.

import { readFileSync, writeFileSync, mkdirSync, readdirSync, copyFileSync, existsSync, rmSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(__dirname, "..");
const repoRoot = resolve(webRoot, "..");
const projectsDir = join(repoRoot, "projects");
const outPdfDir = join(webRoot, "public", "papers");
const outJsonPath = join(webRoot, "src", "data", "publications.json");

const REQUIRED = ["slug", "title", "authors", "year", "topic", "type", "tldr", "pdf"];

function fail(msg) {
  console.error(`[build-manifest] ${msg}`);
  process.exit(1);
}

function listProjectDirs() {
  return readdirSync(projectsDir, { withFileTypes: true })
    .filter((e) => e.isDirectory() && /^\d{2}_/.test(e.name))
    .map((e) => join(projectsDir, e.name))
    .sort();
}

// Reset output PDF dir so removed papers don't linger.
if (existsSync(outPdfDir)) rmSync(outPdfDir, { recursive: true, force: true });
mkdirSync(outPdfDir, { recursive: true });
mkdirSync(dirname(outJsonPath), { recursive: true });

const publications = [];
const seenSlugs = new Set();

for (const projectDir of listProjectDirs()) {
  const metaPath = join(projectDir, "meta.yaml");
  if (!existsSync(metaPath)) {
    console.warn(`[build-manifest] skip ${projectDir} (no meta.yaml)`);
    continue;
  }
  let meta;
  try {
    meta = yaml.load(readFileSync(metaPath, "utf8"));
  } catch (e) {
    fail(`${metaPath}: YAML parse error: ${e.message}`);
  }
  for (const key of REQUIRED) {
    if (meta[key] === undefined || meta[key] === null || meta[key] === "") {
      fail(`${metaPath}: missing required field "${key}"`);
    }
  }
  if (seenSlugs.has(meta.slug)) fail(`${metaPath}: duplicate slug "${meta.slug}"`);
  seenSlugs.add(meta.slug);

  const pdfSrc = join(projectDir, meta.pdf);
  if (!existsSync(pdfSrc)) {
    console.warn(`[build-manifest] ${metaPath}: PDF not found at ${pdfSrc} — card will render without link`);
    publications.push({ ...meta, pdfHref: null });
    continue;
  }
  const pdfDst = join(outPdfDir, `${meta.slug}.pdf`);
  copyFileSync(pdfSrc, pdfDst);
  publications.push({ ...meta, pdfHref: `papers/${meta.slug}.pdf` });
}

publications.sort((a, b) => (b.year - a.year) || a.title.localeCompare(b.title));

writeFileSync(outJsonPath, JSON.stringify({ publications }, null, 2) + "\n");
console.log(`[build-manifest] wrote ${publications.length} publications -> ${outJsonPath}`);
