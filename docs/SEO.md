# SEO audit and publishing guide

Audited the live site on 25 October 2026. This is a technical and on-page audit,
not a ranking forecast. Search Console, field Core Web Vitals, actual reader
subscriptions, and Google's URL Inspection/Rich Results tools were not available
in this session, so no indexing coverage, traffic, or performance score is claimed.

## What was already working

All six authored pages were accessible over HTTPS with canonical URLs. The three
articles already had descriptions and BlogPosting structured data. The sitemap,
robots file, and feed were reachable. These working URLs and feed entry IDs remain
unchanged. The existing Google verification token remains in the shared head.

## Findings and changes

| Finding in the live HTML | Change |
| --- | --- |
| Home, About, and Topics reused one description; homepage title was a generic tagline | Write distinct page descriptions and a more descriptive homepage search title. Allow optional `seo_title` while preserving the visible article title. |
| Article schema and sitemap still reported original publication dates despite visible October 2026 updates | Standardize `last_modified_at` across visible dates, JSON-LD, social metadata, feed, and sitemap. Preserve original publication timestamps. |
| WebSite schema used the author's name for the site's name | Use a linked schema graph with separate WebSite, Person, WebPage/AboutPage, and BlogPosting entities. Keep plain text correctly JSON-escaped. |
| Author schema lacked a profile URL; Twitter creator was `@Yudhiesh Ravindranath` | Link the visible byline and article author to About; use the actual configured handle. About connects the author to their stated OneCredit role. |
| Sharing images lacked alt/dimension metadata; non-post pages had none | Share each post's own generated art with dimensions and alt text; use the homepage mark only for non-post pages. Supply Twitter descriptions as well as Open Graph metadata. |
| Sitemap advertised the raw notebook and 4.5MB chart export | Exclude `assets/` and `src/` exports from sitemap generation. Add `noindex, follow` to those exported HTML files during build preparation; retain direct access and embeds. |
| Article bodies used multiple H1 headings | Keep one H1 supplied by the layout and start article sections at H2. Preserve heading text and link fragments. |
| Article endings offered little contextual discovery | Add automatic links to up to three posts sharing topics, with a small other-writing fallback. |
| Body images generally loaded eagerly, and local Markdown images lacked dimensions | Prepare the generated output to lazy-load body images and add intrinsic dimensions to local PNG/JPEG/WebP images when neither dimension was specified. Keep chapter openers eager. |
| 404 reused the homepage metadata | Add a clear title/description, noindex, and useful recovery links. |

## Shared implementation

`_includes/seo.html` is the single source of title tags, descriptions, canonicals,
robots directives, Open Graph/Twitter tags, verification tags, and JSON-LD.
`_includes/head.html` retains the theme stylesheet, feed discovery, custom head,
and configured analytics. The old SEO tag plugin is no longer invoked, avoiding
duplicate or conflicting tags/structured data.

Descriptions fall back from page `description` to `excerpt` to site description.
A `seo_title` is optional; otherwise the visible title is used. Sharing images
fall back from `image` to article `artwork` to the site mark. The article template
includes explicit image metadata, and output checks verify actual asset sizes.
This is a single-author blog: the shared Person and byline use `site.author`.
Update that author data if the blog gains guest authors.

Keep square art as a square summary card. The current 480×480 illustrations meet
Google's documented minimum image area for Article markup, but do not meet its
1200px width recommendation for large Discover previews. Dedicated 1200px sharing
cards are an optional later design task; no enlarged artwork or Discover eligibility
is claimed here.

`prepare_site.py` changes only generated output, keeping the large exported
notebook/chart files intact in Git. It marks utility HTML noindex and improves
article image loading. Both PR checks and the Pages deployment run it before
`check_seo.py` and `check_site.py`. Local previews should use the README commands
when checking the prepared output. The plain `jekyll serve` preview doesn't apply
the output preparation step.

## For every new article

1. Use a title that identifies the actual problem. Add `seo_title` only if the
   shorter display title needs clarification; keep their meanings aligned.
2. Write a specific, unique description in one or two useful sentences. A character
   count is not a ranking target; Google can choose a different snippet.
3. Include the unique generated artwork, descriptive alt text, and actual image
   dimensions from the template. For external body images, supply accurate width
   and height yourself; build preparation does not fetch third-party assets.
4. Start body headings at H2. Link to relevant existing work when it helps the reader.
   Related-post links are added automatically from your topics.
5. Keep `date` as the real publication timestamp with its timezone. Set
   `last_modified_at` only after a meaningful update; never automatically make
   all articles appear fresh at every build. Legacy `modified_date` is still read
   by the layout/feed, but use `last_modified_at` for new work so the sitemap agrees.
6. Keep published filenames/permalinks stable. A move needs a deliberate redirect;
   changing a title alone should not break an existing URL.
7. Run the build, preparation, and checks, then review the PR. The checks inspect
   all indexable sitemap pages and all article artwork, not just the latest feed.

Do not add keyword meta tags, invented review/FAQ markup, or unsupported claims
for an SEO score. Technical correctness helps discovery and interpretation;
useful, accurate articles and references remain the work that readers come for.

## After merging

- In the existing verified Google Search Console property, submit or recheck
  `https://yudhiesh.github.io/sitemap.xml`.
- Inspect the homepage and one article, and request recrawling after deployment
  if appropriate. Check the article with Google's Rich Results Test.
- Review indexing exclusions and search queries over time. Keep the notebook and
  chart exports crawlable so a crawler can actually see their noindex directives.
- Measure real mobile performance with PageSpeed Insights/CrUX when data exists;
  don't infer Core Web Vitals from HTML size alone.

## Primary references

- [Article structured data](https://developers.google.com/search/docs/appearance/structured-data/article)
- [Title links](https://developers.google.com/search/docs/appearance/title-link)
- [Search snippets](https://developers.google.com/search/docs/appearance/snippet)
- [Sitemap guidance](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap)
- [Noindex and crawler access](https://developers.google.com/search/docs/crawling-indexing/block-indexing)
- [Discover image guidance](https://developers.google.com/search/docs/appearance/google-discover)
