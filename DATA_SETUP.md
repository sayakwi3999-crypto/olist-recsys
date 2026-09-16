# Data loading and preprocessing steps

## 0. the preparing files

- Profiling script: `01_data_profiling.sql`
- Cleaning script: `02_cleaning_and_dedup.sql`

## 1. Load the data into MySQL 

- use navicat or PowerShell/CMD

## 2. Run the profiling script 

Open `01_data_profiling.sql` in Navicat (with the `olist_analysis` database selected) and run it.

It reports row counts per table, duplicate volumes, missing values, dirty rows , referential integrity, and computes below important data: number of users, number of items, repeat-purchase rate and the review-score distribution which in the interaction summary that matters most before modelling.

## 3. Run the cleaning script (builds a clean database)

After running `02_cleaning_and_dedup.sql`, it will:

- create the new database `olist_clean`;
- build 8 clean tables:
  - `clean_orders` (duplicated, missing and anomalous orders dropped)
  - `clean_customers` (one row per user, keeping the most recent account)
  - `clean_order_items` (same order/item/seller merged into a single quantity)
  - `clean_payments` + `order_payment_summary` (order-level payment summary)
  - `clean_reviews`, `clean_products`, `clean_sellers`
- log every removed group of rows into `cleaning_log` and print a comparison report.

`olist_analysis` keeps the raw data untouched, so the script can be run a second time.

## 4. Next step

With the clean database in place, connect from Python to `olist_clean` for feature modelling:

- connection
- interactions: `clean_orders` + `clean_order_items` + `clean_reviews`
- user features: `clean_customers` + order aggregates
- item features: `clean_products` + sales and review aggregates
- context features: time and state from `clean_orders` / `clean_customers`, + delivery timing


## 5. Troubleshooting

- The import takes a while: the 7 dumps total about 45 MB, just let it finish.
