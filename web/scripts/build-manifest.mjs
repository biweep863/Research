#!/usr/bin/env node
// Scans ../projects/*/meta.yaml, validates, copies PDFs into public/papers/,
// and writes src/data/publications.json for the app to consume.
//
// The `updated` date for each publication is derived from git — the commit
// date of the most recent commit that touched anything inside the project
// directory. If git history isn't available (shallow clone, uninitialized
// repo, or working-tree-only changes not yet committed), we fall back to
// the current date so a fresh checkout still produces a valid manifest.

import { readFileSync, writeFileSync, mkdirSync, readdirSync, copyFileSync, existsSync, rmSync } from "node:fs";
import { join, dirname, resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import yaml from "js-yaml";

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(__dirname, "..");
const repoRoot = resolve(webRoot, "..");
const projectsDir = join(repoRoot, "projects");
const outPdfDir = join(webRoot, "public", "papers");
const outJsonPath = join(webRoot, "src", "data", "publications.json");

const REQUIRED = ["slug", "title", "authors", "year", "topic", "type", "tldr", "pdf", "created"];
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

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

function today() {
  return new Date().toISOString().slice(0, 10);
}

// Vercel does a shallow clone by default (depth ~10), which breaks
// `git log` for projects whose most recent commit falls outside that
// window. Unshallow once, best-effort, before we start reading history.
function ensureFullGitHistory() {
  try {
    const isShallow = execFileSync(
      "git",
      ["-C", repoRoot, "rev-parse", "--is-shallow-repository"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] },
    ).trim();
    if (isShallow === "true") {
      execFileSync("git", ["-C", repoRoot, "fetch", "--unshallow", "--quiet"], {
        stdio: "ignore",
      });
    }
  } catch {
    /* no git, no remote, or already full — carry on */
  }
}

function gitLastModified(projectDir) {
  const rel = relative(repoRoot, projectDir);
  try {
    const out = execFileSync(
      "git",
      ["-C", repoRoot, "log", "-1", "--format=%cs", "--", rel],
      { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] },
    ).trim();
    if (DATE_RE.test(out)) return out;
  } catch {
    /* git not available or not a repo */
  }
  return null;
}

ensureFullGitHistory();

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

  // Validate `created` (author-declared, static).
  const createdRaw =
    typeof meta.created === "string" ? meta.created : meta.created?.toISOString?.().slice(0, 10);
  if (!createdRaw || !DATE_RE.test(createdRaw)) {
    fail(`${metaPath}: field "created" must be YYYY-MM-DD, got "${meta.created}"`);
  }
  meta.created = createdRaw;

  // Derive `updated` from git; fall back to today (fresh checkout, uncommitted).
  meta.updated = gitLastModified(projectDir) ?? today();

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
