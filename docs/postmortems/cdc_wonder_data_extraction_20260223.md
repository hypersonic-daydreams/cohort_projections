---
date: 2026-02-23T09:35:00-06:00
title: Post-Mortem: CDC WONDER Natality Data Extraction
author: Antigravity AI
files_created:
  - data/raw/fertility/cdc_wonder_nd_births_2020_2023.txt
  - data/raw/fertility/cdc_wonder_national_births_2020_2023.txt
---

# Post-Mortem: CDC WONDER Data Extraction

## Background
The objective was to extract Single Race 6 categories from the expanded CDC WONDER Natality dataset (2016-2024) for North Dakota (2020-2023) and the United States (2020-2023). 

## What Went Wrong
The extraction process encountered significant delays and required extensive trial-and-error due to the following factors:

1. **Browser Subagent vs. Host Environment Discordance:**
   - Initially, a specialized browser subagent was used to navigate the form and trigger the download. This succeeded, but because the subagent operates in an isolated container environment, the downloaded `.txt` files were saved to the container's virtual filesystem rather than the host Windows `Downloads` folder or the WSL `/home` directory.
   - Considerable time was lost attempting to locate these "ghost" files in the host's Windows and WSL filesystems before the container isolation issue was fully diagnosed.

2. **Playwright Headless Browser Timeouts:**
   - To bypass the container issue, a local Python Playwright script was implemented.
   - The CDC WONDER form relies on a complex, dynamic UI with hidden dropdowns and non-standard identifiers. Form selectors had to be rewritten multiple times to accurately target the "Mother's Age 9" and "Mother's Single Race 6" elements.
   - The queries themselves (especially the National query) take several minutes to run on CDC's servers. The default Playwright timeouts (30 seconds) caused scripts to fail prematurely, requiring multiple runs with progressively longer timeouts (up to 10 minutes).

3. **Dynamic UI Changes Triggering Download Failures:**
   - Even after the query successfully completed, CDC WONDER's new UI hides the download link behind an "Export" toggle menu. In headless Chrome, attempting to attach to the `download` event via the Export button persistently timed out.

## What Finally Worked
The automated download event listener was abandoned in favor of a hybrid scraping approach:
1. **Headless Form Submission:** Playwright was used to navigate the form, fill in the exact grouping criteria, and click "Send".
2. **Raw HTML Extraction:** Instead of attempting to trigger and intercept the `.txt` download, the script waited for the results table to load on the page and then captured the raw, fully-rendered HTML of the results page.
3. **Local Parsing:** A secondary Python script utilizing `BeautifulSoup` parsed the extracted HTML tables (`response-form`) row-by-row and wrote the data out directly to the requested `.txt` files in tab-separated format (`tsv`), faithfully reproducing the expected CDC WONDER export layout.

## Cleanup Plan executed
During the troubleshooting process, a large number of temporary scripts, virtual environments, and HTML dumps were created in `/tmp`. The following cleanup steps were executed to remove the clutter:

1. Removed the temporary Python virtual environment: `rm -rf /tmp/wonder_venv/`
2. Removed exploratory and final Playwright scrape scripts: `rm -f /tmp/*wonder*.py`
3. Removed raw HTML dumps: `rm -f /tmp/wonder_results.html /tmp/wonder_national_results.html`
4. Removed debugging screenshots: `rm -f /tmp/wonder*.png`
