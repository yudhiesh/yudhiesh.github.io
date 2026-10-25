"""Check generated navigation and feed compatibility, without external requests."""
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '_site')
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    feed = ET.parse(root / 'feed.xml').getroot()
    require(feed.findtext('a:title', namespaces=ns) == 'Bits & Backprops', 'Feed title was escaped incorrectly')
    entries = feed.findall('a:entry', ns)
    require(entries, 'Feed has no posts')
    dates = [datetime.fromisoformat(e.findtext('a:published', namespaces=ns)) for e in entries]
    require(dates == sorted(dates, reverse=True), 'Feed posts are not newest first')
    ids = [e.findtext('a:id', namespaces=ns) for e in entries]
    require(len(ids) == len(set(ids)), 'Feed has duplicate entry IDs')
    for entry in entries:
        url = entry.find('a:link', ns).attrib['href']
        article = root / unquote(urlparse(url).path).lstrip('/') / 'index.html'
        require(article.is_file(), f'Feed points to a missing article: {url}')
        page = Page(article.read_text())
        page_ids = {a['id'] for _, a in page.elements if 'id' in a}
        for tag, attrs in page.elements:
            href = attrs.get('href', '')
            if tag == 'a' and href.startswith('#'):
                require(unquote(href[1:]) in page_ids, f'Broken article anchor: {url}{href}')
            src = attrs.get('src', '')
            if tag == 'img' and src.startswith('/'):
                require((root / unquote(src).lstrip('/')).is_file(), f'Missing image: {src}')
        portable = Page(entry.findtext('a:content', namespaces=ns))
        require(any(t == 'img' and '/images/art/' in a.get('src', '') for t, a in portable.elements), f'Feed artwork missing: {url}')
        require(not any(t in {'script', 'iframe', 'video'} for t, _ in portable.elements), f'Interactive-only markup in feed: {url}')
        for tag, attrs in portable.elements:
            for key in ('src', 'href'):
                if key in attrs:
                    require(not attrs[key].startswith(('/', '#')), f'Relative feed URL: {attrs[key]}')
        if 'cerberus-safeguards' in url:
            require(any(t == 'img' and '/diagrams/synthetic-data-pipeline.png' in a.get('src', '') for t, a in portable.elements), 'Static pipeline missing from feed')
            require(any(t == 'a' and a.get('href', '').endswith('/assets/eda.html') for t, a in portable.elements), 'Feed analysis fallback missing')
    expected_ids = {
        'https://yudhiesh.github.io/2025/06/16/cerberus-safeguards-for-llms-synthetic-data-generation-part-1',
        'https://yudhiesh.github.io/2025/05/09/the-achilles-heel-of-vector-search-filters',
        'https://yudhiesh.github.io/2024/12/27/feature-flag-driven-development-in-ml',
    }
    for expected in expected_ids:
        path = root / urlparse(expected).path.lstrip('/') / 'index.html'
        require(path.is_file(), f'Existing article URL changed: {expected}')
    for entry in entries:
        url = entry.find('a:link', ns).attrib['href'].rstrip('/')
        if url in expected_ids:
            require(entry.findtext('a:id', namespaces=ns) == url, f'Existing feed entry ID changed: {url}')
    topics = Page((root / 'tags.html').read_text())
    topic_ids = {a['id'] for _, a in topics.elements if 'id' in a}
    for slug in ('llm-safety', 'synthetic-data', 'search', 'ml-engineering', 'vector-search', 'machine-learning', 'cerberus'):
        require(slug in topic_ids, f'Topic or legacy alias missing: {slug}')
    about = (root / 'about/index.html').read_text()
    require('Lead DS' in about and 'https://portal.onecredit.my/' in about, 'About page has stale role/link')
    print(f'PASS: {len(entries)} feed entries, stable IDs/URLs, artwork, static fallbacks, article anchors, topics, and current role')


if __name__ == '__main__':
    main()
