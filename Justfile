# Justfile for Jekyll GitHub Pages blog in docs/ directory

# Default command - show available commands
default:
    @just --list

# Install Jekyll and dependencies
install:
    cd docs && bundle install

# Update dependencies
update:
    cd docs && bundle update

# Serve the site locally with live reload
serve:
    cd docs && bundle exec jekyll serve --livereload

# Serve with drafts visible
serve-drafts:
    cd docs && bundle exec jekyll serve --livereload --drafts

# Serve on a different port (useful if 4000 is taken)
serve-port port="4001":
    cd docs && bundle exec jekyll serve --livereload --port {{port}}

# Build the site
build:
    cd docs && bundle exec jekyll build

# Build for production (with JEKYLL_ENV=production)
build-prod:
    cd docs && JEKYLL_ENV=production bundle exec jekyll build

# Clean the site (remove _site directory)
clean:
    cd docs && bundle exec jekyll clean

# Check for broken links and HTML issues
check:
    cd docs && bundle exec htmlproofer ./_site --disable-external

# Create a new post with today's date
new-post title:
    #!/usr/bin/env bash
    date=$(date +%Y-%m-%d)
    filename="docs/_posts/${date}-{{title}}.md"
    mkdir -p docs/_posts
    echo "---" > $filename
    echo "layout: post" >> $filename
    echo "title: \"{{title}}\"" >> $filename
    echo "date: ${date} $(date +%H:%M:%S) +0000" >> $filename
    echo "categories: " >> $filename
    echo "---" >> $filename
    echo "" >> $filename
    echo "Post created: $filename"

# Create a new draft
new-draft title:
    #!/usr/bin/env bash
    mkdir -p docs/_drafts
    filename="docs/_drafts/{{title}}.md"
    echo "---" > $filename
    echo "layout: post" >> $filename
    echo "title: \"{{title}}\"" >> $filename
    echo "categories: " >> $filename
    echo "---" >> $filename
    echo "" >> $filename
    echo "Draft created: $filename"

# Open the site in default browser
open:
    open http://localhost:4000

# Serve and open in browser
dev: 
    just serve &
    sleep 3
    just open

# Full rebuild - clean, install deps, and serve
fresh:
    just clean
    just install
    just serve

# Watch for changes and rebuild (without serving)
watch:
    cd docs && bundle exec jekyll build --watch

# List all posts
list-posts:
    @ls -la docs/_posts/ 2>/dev/null || echo "No posts found"

# List all drafts
list-drafts:
    @ls -la docs/_drafts/ 2>/dev/null || echo "No drafts found"

# Count posts and drafts
stats:
    @echo "Posts: $(ls docs/_posts/*.md 2>/dev/null | wc -l)"
    @echo "Drafts: $(ls docs/_drafts/*.md 2>/dev/null | wc -l)"

# Validate _config.yml
validate-config:
    cd docs && bundle exec jekyll doctor

# Check Ruby and Bundler versions
check-env:
    @echo "Ruby version:"
    @ruby --version
    @echo "\nBundler version:"
    @bundle --version
    @echo "\nJekyll version:"
    @cd docs && bundle exec jekyll --version

# Add webrick if needed (for Ruby 3.0+)
add-webrick:
    cd docs && bundle add webrick

# Quick setup for common missing dependencies
setup-deps:
    cd docs && bundle add webrick
    cd docs && bundle install

# Kill the Jekyll server running on default port
kill:
    @lsof -ti:4000 | xargs kill -9 2>/dev/null || echo "No process found on port 4000"

# Kill Jekyll server on a specific port
kill-port port:
    @lsof -ti:{{port}} | xargs kill -9 2>/dev/null || echo "No process found on port {{port}}"

# Find what's running on port 4000
find-port:
    @lsof -i:4000 || echo "Port 4000 is free"

# Kill all Ruby processes (use with caution)
kill-all-ruby:
    @pkill -f ruby || echo "No Ruby processes found"
