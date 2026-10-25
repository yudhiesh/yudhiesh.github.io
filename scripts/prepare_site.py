"""Prepare Jekyll output: keep utility exports out of search; size/lazy-load local media."""
import re
import struct
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


class ImageTag(HTMLParser):
    def handle_starttag(self, tag, attrs):
        self.attrs = dict(attrs)


def image_size(path):
    data = path.read_bytes()
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return struct.unpack('>II', data[16:24])
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        kind = data[12:16]
        if kind == b'VP8X':
            return int.from_bytes(data[24:27], 'little') + 1, int.from_bytes(data[27:30], 'little') + 1
        if kind == b'VP8L':
            bits = int.from_bytes(data[21:25], 'little')
            return (bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1
        signature = data.find(b'\x9d\x01\x2a', 20, 40)
        if signature >= 0:
            w, h = struct.unpack('<HH', data[signature + 3:signature + 7])
            return w & 0x3fff, h & 0x3fff
    if data.startswith(b'\xff\xd8'):
        i = 2
        while i + 8 < len(data):
            if data[i] != 0xff:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker == 0xff:
                i -= 1
                continue
            if marker in {0xd8, 0xd9}:
                continue
            length = int.from_bytes(data[i:i + 2], 'big')
            if marker in {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}:
                height, width = struct.unpack('>HH', data[i + 3:i + 7])
                return width, height
            if length < 2:
                break
            i += length
    return None


def prepare(root):
    utilities = 0
    for directory in ('assets', 'src'):
        for path in (root / directory).rglob('*.html'):
            source = path.read_text()
            if not re.search(r'<meta\b[^>]*name=[\"\']robots[\"\']', source, re.I):
                source, count = re.subn(r'(<head\b[^>]*>)', r'\1\n<meta name="robots" content="noindex, follow">', source, count=1, flags=re.I)
                if count != 1:
                    raise ValueError(f'Utility export has no head: {path}')
                path.write_text(source)
            utilities += 1
    media = 0
    for path in root.rglob('*.html'):
        if path.parts[len(root.parts)] in ('assets', 'src'):
            continue
        source = path.read_text()
        if 'h-entry' not in source:
            continue

        def prepare_image(match):
            nonlocal media
            original = match.group(0)
            parser = ImageTag()
            parser.feed(original)
            attrs = parser.attrs
            if attrs.get('fetchpriority') == 'high':
                return original
            extra = ''
            if 'loading' not in attrs:
                extra += ' loading="lazy"'
            if 'decoding' not in attrs:
                extra += ' decoding="async"'
            src = attrs.get('src', '')
            if src.startswith('/') and 'width' not in attrs and 'height' not in attrs:
                asset = root / unquote(urlparse(src).path).lstrip('/')
                if asset.is_file():
                    size = image_size(asset)
                    if size:
                        extra += f' width="{size[0]}" height="{size[1]}"'
            if not extra:
                return original
            media += 1
            closing = '/>' if original.endswith('/>') else '>'
            return original[:-len(closing)] + extra + closing

        source = re.sub(r'''<img\b(?:[^>"']|"[^"]*"|'[^']*')*>''', prepare_image, source, flags=re.I)
        path.write_text(source)
    print(f'Prepared {utilities} noindex utility exports and {media} article images')


if __name__ == '__main__':
    prepare(Path(sys.argv[1] if len(sys.argv) > 1 else '_site'))
