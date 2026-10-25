"""Validate rendered SEO contracts for every indexable sitemap page and article."""
import json
import re
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET
from prepare_site import image_size


class Document(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.title = ''
        self.h1 = 0
        self.schemas = []
        self.in_title = False
        self.in_schema = False
        self.schema_text = ''
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs))
        if tag == 'title':
            self.in_title = True
        if tag == 'h1':
            self.h1 += 1
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self.in_schema = True
            self.schema_text = ''

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self.in_schema:
            self.schema_text += data

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        if tag == 'script' and self.in_schema:
            self.schemas.append(json.loads(self.schema_text))
            self.in_schema = False

    def meta(self, name):
        return [a.get('content', '') for t, a in self.elements if t == 'meta' and (a.get('name') == name or a.get('property') == name)]


def require(value, message):
    if not value:
        raise AssertionError(message)


def local_file(root, url):
    path = root / unquote(urlparse(url).path).lstrip('/')
    return path / 'index.html' if urlparse(url).path.endswith('/') else path


def check(root):
    ns = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    rows = ET.parse(root / 'sitemap.xml').getroot().findall('s:url', ns)
    urls = [row.findtext('s:loc', namespaces=ns) for row in rows]
    require(len(urls) == len(set(urls)), 'Duplicate sitemap URLs')
    core_pages = {'https://yudhiesh.github.io/', 'https://yudhiesh.github.io/about/', 'https://yudhiesh.github.io/tags.html'}
    require(core_pages.issubset(urls), 'A core page is missing from the sitemap')
    descriptions, titles = set(), set()
    article_count = 0
    for row in rows:
        url = row.findtext('s:loc', namespaces=ns)
        require(url.startswith('https://yudhiesh.github.io/'), f'Noncanonical sitemap URL: {url}')
        require(not any(part in urlparse(url).path for part in ('/assets/', '/src/', '/404.html', '/feed.xml')), f'Utility export leaked into sitemap: {url}')
        source = local_file(root, url).read_text()
        page = Document(source)
        require(page.title.strip() and page.title not in titles, f'Missing/duplicate title: {url}')
        titles.add(page.title)
        desc = page.meta('description')
        require(len(desc) == 1 and len(desc[0]) >= 60 and desc[0] not in descriptions, f'Missing/duplicate description: {url}')
        descriptions.add(desc[0])
        canonicals = [a.get('href') for t, a in page.elements if t == 'link' and a.get('rel') == 'canonical']
        require(canonicals == [url], f'Wrong/duplicate canonical: {url}')
        require(page.h1 == 1, f'Expected one primary H1: {url}')
        require(not any('noindex' in value for value in page.meta('robots')), f'Article/page accidentally noindexed: {url}')
        require(page.meta('og:url') == [url], f'Sharing URL differs from canonical: {url}')
        require(page.meta('twitter:creator') == ['@yudhiesh1997'], f'Malformed creator handle: {url}')
        images = page.meta('og:image')
        require(len(images) == 1 and images[0].startswith('https://'), f'Missing/duplicate sharing image: {url}')
        image = local_file(root, images[0])
        require(image.is_file(), f'Missing sharing asset: {images[0]}')
        require(page.meta('og:image:alt') and len(page.meta('og:image:alt')[0]) > 15, f'Missing image description: {url}')
        actual_size = image_size(image)
        declared_size = (int(page.meta('og:image:width')[0]), int(page.meta('og:image:height')[0]))
        require(actual_size == declared_size, f'Incorrect image dimensions: {url}')
        require(len(page.schemas) == 1, f'Duplicate/missing schema graph: {url}')
        graph = page.schemas[0]['@graph']
        website = next(n for n in graph if n['@type'] == 'WebSite')
        require(website['name'] == 'Bits & Backprops', 'Wrong site identity')
        articles = [n for n in graph if n['@type'] == 'BlogPosting']
        if not articles:
            continue
        article_count += 1
        require(len(articles) == 1, f'Duplicate article schema: {url}')
        article = articles[0]
        require(article['url'] == url and article['image']['url'] == images[0], f'Inconsistent article metadata: {url}')
        require(article['author']['url'] == 'https://yudhiesh.github.io/about/', f'Author profile missing: {url}')
        require(article['dateModified'] == row.findtext('s:lastmod', namespaces=ns), f'Sitemap/schema dates disagree: {url}')
        require(page.meta('article:modified_time') == [article['dateModified']], f'Sharing/schema dates disagree: {url}')
        require(datetime.fromisoformat(article['dateModified']) >= datetime.fromisoformat(article['datePublished']), f'Update precedes publication: {url}')
        visible_modified = [a.get('datetime') for t, a in page.elements if t == 'time' and 'dt-modified' in a.get('class', '').split()]
        require(not visible_modified or visible_modified == [article['dateModified']], f'Visible/schema dates disagree: {url}')
        require(any(t == 'a' and a.get('href') == '/about/' for t, a in page.elements), f'Author profile unlinked: {url}')
        for tag, attrs in page.elements:
            if tag == 'img' and attrs.get('fetchpriority') != 'high':
                require(attrs.get('loading') == 'lazy', f'Body image is eagerly loaded: {url}')
            if tag == 'a' and attrs.get('href', '').startswith('/'):
                path = attrs['href'].split('#')[0]
                if path:
                    require(local_file(root, path).is_file(), f'Broken internal link: {path}')
    require(article_count >= 3, 'Articles missing from sitemap audit')
    for directory in ('assets', 'src'):
        for export in (root / directory).rglob('*.html'):
            require(any('noindex' in value for value in Document(export.read_text()).meta('robots')), f'Utility export is indexable: {export}')
    require(any('noindex' in v for v in Document((root / '404.html').read_text()).meta('robots')), '404 page is indexable')
    require('Sitemap: https://yudhiesh.github.io/sitemap.xml' in (root / 'robots.txt').read_text(), 'Robots sitemap discovery missing')
    print(f'PASS SEO: {len(urls)} indexable pages, {article_count} articles; titles, descriptions, canonicals, schema, dates, images, links, and noindex exports')


if __name__ == '__main__':
    check(Path(sys.argv[1] if len(sys.argv) > 1 else '_site'))
