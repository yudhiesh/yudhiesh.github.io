# Bits & Backprops

A Jekyll notebook on ML and engineering, published at https://yudhiesh.github.io.
The site source lives in `docs/`; GitHub Pages deploys `master` through the existing workflow.

## Preview locally

With Ruby and Bundler installed:

```sh
cd docs
bundle install
bundle exec jekyll serve
```

Open http://localhost:4000. Build without serving with `bundle exec jekyll build`.
Do not edit `_site/`: it is generated output.

## Write a post

Copy `templates/post.md` into `docs/_posts/YYYY-MM-DD-your-title.md`, replace the
sample front matter, and write in Markdown. Existing URLs are controlled by the
permalink in `docs/_config.yml`; keep published filenames stable.

Every post needs its own generated engraving. Generate a subject connected to
the article, then set `artwork` and a descriptive `artwork_alt`. Shared opener
images are rejected by the PR checks. The ginkgo belongs to the homepage only.
`image` independently controls the social preview. Use a site-relative asset path, such as
`/images/art/your-subject.webp`, and a meaningful alt description. Optional
`artwork_caption`, `subtitle`, `series`, `part`, and `last_modified_at` fields are
supported. The layout adds the date, reading time, author, topic links, and return
link automatically.

See [the editorial guide](docs/EDITORIAL.md) for the art direction, reusable art
prompt, writing style, and a small pre-publish checklist.

## Validate a build

After building from `docs/`, run these from the repository root:

```sh
python3 scripts/prepare_site.py docs/_site
python3 scripts/check_seo.py docs/_site
python3 scripts/check_site.py docs/_site
```

Both PR checks and production deployment run the same preparation and checks.
This checks the feed as XML, entry IDs and URLs, portable artwork and diagrams,
article anchors, topic aliases, the current About link, and unique artwork on every post.

See [the SEO audit and publishing guide](docs/SEO.md) for metadata defaults,
search exclusions, modification dates, and post-publication checks.
