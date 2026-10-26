# The notebook's editorial direction

## Visual direction

The site uses the color and scale of vintage science fiction cover art: midnight
blue, teal, ochre light, and rust red. The homepage is a cover, with a wide image,
large title, and a small catalogue of illustrated stories. Each article opens with
its own scene. Long-form text stays on warm paper so technical writing remains easy
to read.

The current scenes came from the user-supplied images in `docs/images/`: the
luminous tree for the homepage, the crimson temple for Cerberus, the eye over
dunes for vector search, and the rocket launch for feature flags. The site serves
optimized WebP copies from `docs/images/art/`. The source PNGs do not need to be
published. The older indigo engravings remain in the repository as a record of the
previous design.

For future articles, choose one image connected to the post and export a unique,
optimized landscape WebP. Add `artwork`, descriptive `artwork_alt`, and an `image`
object with its real width, height, and alt text for sharing cards. Check the
scene at both full width and card size. Never put essential information only in
art, and credit any third-party material that requires attribution.

## Writing and navigation

Write like you are talking to a curious engineer over coffee. Use first person
when describing your own work, name the actual problem, and explain the result
before listing tools. Prefer short paragraphs and concrete examples. Include
what failed and what remains uncertain.

Use one page title and start article sections at `##`; the layout supplies the
H1. Keep code, diagrams, tables, equations, and embeds readable on phones. Use
`templates/post.md` instead of copying layout markup into articles.

Use the curated topic slugs in `_data/topics.yml`: `llm-safety`, `synthetic-data`,
`search`, and `ml-engineering`. One or two topics per post is usually enough.
Historical tag anchors remain as aliases so older shared links still work.

`feed.xml` keeps existing Atom endpoints and entry IDs, includes the post artwork,
and publishes full articles with absolute links. Keep JavaScript in the layout,
not Markdown. Put a static diagram and direct link alongside interactive material;
the feed removes scripts, frames, and video while keeping explanatory text.

## Before publishing

- Replace the template's title, date, excerpt, description, and tags.
- Add unique artwork with meaningful alt text and accurate social image dimensions.
- Preview the homepage, article, and topic links at desktop and phone widths.
- Check keyboard focus, code overflow, equations, and embedded notebooks.
- Use `last_modified_at` only for a real update. See `SEO.md` for metadata defaults.
- Build with `bundle exec jekyll build` from `docs/`, then run the preparation and
  checks in the repository README.
- Open a PR, review its diff and CI checks, then merge when ready to publish.
