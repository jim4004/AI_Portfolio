# Cereal Pricing Analysis (Dominick's Scanner Data)

A practical pricing exercise using the public Dominick's Finer Foods cereal category data from the Kilts Center at the University of Chicago Booth School of Business.

## Overview

This notebook loads store-level weekly scanner data for the cereal category, cleans it, aggregates to weekly product and category series, and estimates a simple price elasticity. It then calculates a revenue-oriented recommended price given a user-supplied target margin.

The analysis is performed at the category and individual UPC level using a time-based holdout. It is intended as a portfolio example of turning retail scanner data into a pricing insight.

## Data Source

- **Source**: James M. Kilts Center for Marketing, University of Chicago Booth School of Business  
- **Dataset**: Dominick's Finer Foods scanner data (cereal category)  
- **Period**: Approximately 1989–1994  
- **Files used**:
  - `upccer.csv` — product descriptions
  - `wcer.csv` — weekly store-level movement (price, units, etc.)

These files are **not** included in the repository (they exceed GitHub's file size limit). Download them from:  
https://www.chicagobooth.edu/research/kilts/research-data/dominicks

Place the CSVs in a local `data/` folder before running the notebook.

**Acknowledgment**: Data provided by the Kilts Center for Marketing at the University of Chicago Booth School of Business. These data are for academic research purposes only.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib jupyter ipykernel
