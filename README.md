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

All posts get a small ginkgo engraving above the title by default. To use a
post-specific illustration, add `artwork` and `artwork_alt`. `image` independently
controls the social preview. Use a site-relative asset path, such as
`/images/art/your-subject.webp`, and a meaningful alt description. Optional
`artwork_caption`, `subtitle`, `series`, `part`, and `modified_date` fields are
supported. The layout adds the date, reading time, author, topic links, and return
link automatically.

See [the editorial guide](docs/EDITORIAL.md) for the art direction, reusable art
prompt, writing style, and a small pre-publish checklist.
