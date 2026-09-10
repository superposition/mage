# Mage documentation

GitHub Pages source for https://superposition.github.io/mage/.

Run `bundle install`, then `bundle exec jekyll serve --baseurl /mage` in this directory.
Open http://localhost:4000/mage/. The docs build requires no GPU or Python dependencies.

Public field notes explain the mathematical ideas and their significance.
Setup instructions (`guide.md`), profiler reference (`profiling.md`), validation
records, and raw result assets remain in GitHub and are excluded from Pages.

Keep the shared experiment ID and a reciprocal link to the Superposition journal.
Pushes to master deploy; pull requests build a preview artifact.

Profile figures are generated from committed experiment evidence with
`uv run --script scripts/plot-comparison.py` from the repository root. See
`experiments/mage-001-comparison.md` for measurement and regeneration commands.
Matplotlib is only needed to regenerate figures; the Pages build uses the
committed SVG/PNG files. Keep short explanations beside public figures and the
full methodology in the GitHub record.
