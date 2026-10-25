# The notebook's editorial direction

## The plan

1. Establish a quiet, book-like reading surface: warm paper, charcoal text, indigo
   links, generous space, and a roughly 700px text column. Use system Georgia for
   literary type without a font download. Keep metadata and navigation in a small
   system sans serif. Use solid backgrounds and thin rules rather than cards.
2. Open every post with its own generated engraving, with a subject tied to the
   article. Keep the indigo ink and technique consistent, but never reuse an
   opener between posts. The ginkgo sprig is the homepage mark only.
3. Keep a simple chronological homepage. Titles and short, useful excerpts should
   do the work. Topics and RSS remain easy to find. Keep diagrams, code, tables,
   and embeds readable on phones.
4. Publish through the existing Markdown → PR → GitHub Pages workflow. Use the
   post template rather than hand-copying layout markup into every article.
5. For the next posts, start with one real question from your work. Keep a small
   backlog of questions, experiments, and surprises; publish when there is a
   useful result rather than padding a schedule. A follow-up to Cerberus can
   begin with the classifier experiment and the errors worth investigating.

Inspired by the quiet chapter-opener format at https://www.k-a.in/AttnRes.html.
The illustrations, copy, and implementation here are original; no source art is reused.

## Art direction

Antique natural-history engravings: delicate crosshatching, muted indigo ink,
transparent backgrounds, and one clear subject. Small, centered, and uncropped
above the title, not a full-width hero. Avoid text in images. Start with a square
canvas, leave generous margins, and export a transparent WebP around 480px square.
Keep the same ink and technique as the subject changes between posts. Aim for
under 100KB per opener; check its silhouette at the actual 160–190px display size.

Reusable image prompt:

> An original editorial chapter-opener engraving of [one subject connected to
> the post]. Antique natural-history copperplate etching, fine crosshatching
> and stippling, muted indigo ink. Quiet, thoughtful, dignified. A single centered
> subject on a square transparent canvas, generous negative space, whole subject
> visible, legible at 160 pixels wide. No frame, text, watermark, or background scene.

Current artwork was created with the built-in image generator: a gentle
three-headed dog guarding a flowering branch for Cerberus, an antique sieve
holding pebbles for vector filtering, and a small sailing boat with a signal
pennant for shipping with feature flags. The homepage uses a ginkgo sprig.
Assets live in `docs/images/art/`. The full prompts for the new search and
feature-flags engravings are recorded in `_data/artwork.json`. Add a credit in `artwork_caption` if future art
requires attribution. Do not hotlink someone else's illustration.

## Voice

Write like you are talking to a curious engineer over coffee. Use first person
when describing your own work, name the actual problem, and explain the result
before listing tools. Prefer short paragraphs and concrete examples. A small
aside is welcome; a joke in every heading is not required.

Avoid launch-style hype, shouted emphasis, and generic claims such as “robust,”
“revolutionary,” or “production-ready” without evidence. Include what failed,
what remains uncertain, and the circumstances under which the result applies.
A reader should feel invited to think with you.

The Cerberus introduction is the first example of this voice. Its remaining
technical sections and older posts can be edited gradually as their experiments
are revisited; they do not need a wholesale rewrite to use the new presentation.

## Before publishing

- Replace the template's title, date, excerpt, description, and tags.
- Generate a unique opener for the article and set `artwork` and `artwork_alt`.
  The build checks that each post has descriptive alt text and its own image
  asset/content, including posts outside the feed window.
- Use `artwork_alt` to describe new art. Keep `image` separate from `artwork` when
  the sharing image should differ. Never put essential information only in art.
- Preview the homepage, article, and its topic links. Check a narrow phone view,
  keyboard focus, code overflow, equations, and any embedded notebooks.
- Run `bundle exec jekyll build` from `docs/`.
- Open a PR, review the diff, and merge when ready to publish.

## Topics and feed compatibility

Use the curated topic slugs in `_data/topics.yml`: `llm-safety`, `synthetic-data`,
`search`, and `ml-engineering`. One or two topics per post is usually enough.
Add a new topic only when it helps readers find a distinct body of work. Historical
tag anchors are retained as aliases so older shared links still land in a useful section.

`feed.xml` keeps the existing Atom endpoint and entry IDs, adds the chapter artwork,
and publishes the full article with absolute links. Keep JavaScript in the layout,
not the Markdown body. Put a static diagram and a direct link alongside interactive
material; the feed removes scripts, frames, and video while keeping their explanatory text.
The original static diagram sources live in `images/diagrams/` beside their PNG exports.
