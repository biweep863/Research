# Papers site

Static Next.js catalog for the papers in `../projects/*/`.
Auto-updates when you push a new project.

## How it stays in sync

Every project directory under `../projects/NN_name/` has a
`meta.yaml` and a `paper/main.pdf`. On each build, the
`scripts/build-manifest.mjs` script:

1. scans `../projects/*/meta.yaml`,
2. copies `paper/main.pdf` → `public/papers/<slug>.pdf`,
3. writes `src/data/publications.json` for the app.

The site renders the JSON with client-side filters.

## Local development

```bash
cd web
npm install
npm run dev           # http://localhost:3000
```

The `predev` hook runs the manifest step, so newly added
`meta.yaml`s show up on the next reload.

## Local production build

```bash
npm run build
npm start          # serves the built app on http://localhost:3000
```

## Deploy (Vercel)

The site is deployed on Vercel. Vercel's GitHub integration
auto-builds and deploys on every push to `main` — no workflow
or CI file needed in this repo.

Project settings on Vercel:
- **Root Directory:** `web`
- **Framework Preset:** Next.js (auto-detected)
- **Build Command:** default (`npm run build`) — the `prebuild`
  hook runs `scripts/build-manifest.mjs` automatically.

The manifest script self-heals a shallow clone (`git fetch --unshallow`)
so the git-derived `updated` date is accurate even in Vercel's
default shallow checkout.

## Adding a new paper

```bash
mkdir -p ../projects/07_new_topic/paper
cp mypaper.pdf ../projects/07_new_topic/paper/main.pdf
cat > ../projects/07_new_topic/meta.yaml <<'YAML'
slug: new_topic
title: "My paper title"
authors: ["Luis Benvenuto"]
year: 2027
topic: "Attention"
type: "Preprint"
venue: "arXiv preprint"
tldr: "One-sentence hook."
pdf: "paper/main.pdf"
created: "2027-01-15"
YAML
git add ../projects/07_new_topic
git commit -m "Add project 07"
git push          # Vercel redeploys the site automatically
```

## meta.yaml schema

| field   | type       | required | notes                                        |
|---------|------------|----------|----------------------------------------------|
| slug    | string     | yes      | unique; used as `/papers/<slug>.pdf`         |
| title   | string     | yes      |                                              |
| authors | string[]   | yes      |                                              |
| year    | number     | yes      |                                              |
| topic   | string     | yes      | drives badge color (see `src/lib/utils.ts`)  |
| type    | string     | yes      | Preprint / Workshop / Journal / Conference   |
| venue   | string     | no       | free text; shown in card footer              |
| tldr    | string     | yes      | one-sentence hook (1–2 lines)                |
| pdf     | string     | yes      | path to PDF relative to the project dir      |
| created | YYYY-MM-DD | yes      | date the paper was first added (static)      |

The `updated` date is **not** in `meta.yaml` — it is derived on every build
from `git log -1 --format=%cs -- projects/NN_name/`, so it bumps
automatically whenever any file inside the project directory is committed.
The GitHub Actions workflow checks out with `fetch-depth: 0` so full
history is available. Locally, if the working tree hasn't been committed
yet, the script falls back to today's date.

To add a new topic color, edit the `TOPIC_COLORS` map in
`src/lib/utils.ts`.
