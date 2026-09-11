# Data loading and preprocessing steps

## 0. Where things live

- Raw data (7 SQL dumps): the local `data/` folder next to this repository (git-ignored).
- Profiling script: `01_data_profiling.sql`
- Cleaning script: `02_cleaning_and_dedup.sql`
- Local MySQL 8.0 (port 3306, Windows service `MySQL80`)

## 1. Load the data into MySQL (pick one route)

### Route A: Navicat (easiest)

1. Open Navicat and connect to the local MySQL server (localhost / 3306 / root / your password).
2. Right-click the connection -> New Database: name it `olist_analysis`, charset `utf8mb4`.
3. Open the new database -> right-click -> Run SQL File -> pick the 7 dumps one by one.
   Suggested order: customers -> sellers -> products -> orders -> order_items -> payments -> reviews.
4. When finished, run `SELECT COUNT(*) FROM orders;` — it should return **99,441**, which confirms the import.

### Route B: command line (PowerShell or CMD)

1. Store the credentials once (you will be prompted for the password):
   `"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql_config_editor.exe" set --login-path=local --host=localhost --user=root --password`
2. Create the database:
   `"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" --login-path=local -e "CREATE DATABASE IF NOT EXISTS olist_analysis DEFAULT CHARACTER SET utf8mb4;"`
3. Import the 7 files one by one:
   `"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" --login-path=local olist_analysis < "data\olist_analysis_customers.sql"`
   (repeat for the remaining six files, changing only the file name)

> On Windows PowerShell 5.1, piping SQL that contains Chinese text into `mysql` can mangle the
> characters. Setting `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` first avoids it.

## 2. Run the profiling script (read-only)

Open `01_data_profiling.sql` in Navicat (with the `olist_analysis` database selected) and run it.

It reports row counts per table, duplicate volumes, missing values, dirty rows (impossible timestamp
orders, negative prices, invalid review scores), referential integrity, and the interaction summary that
matters most before modelling: number of users, number of items, repeat-purchase rate and the review-score
distribution.

## 3. Run the cleaning script (builds a clean database; the raw one is untouched)

Run `02_cleaning_and_dedup.sql`. It will:

- create the new database `olist_clean`;
- build 8 clean tables:
  - `clean_orders` (de-duplicated, missing and anomalous orders dropped)
  - `clean_customers` (one row per user, keeping the most recent account)
  - `clean_order_items` (same order/item/seller merged into a single quantity)
  - `clean_payments` + `order_payment_summary` (order-level payment summary)
  - `clean_reviews`, `clean_products`, `clean_sellers`
- log every removed group of rows into `cleaning_log` and print a before/after comparison report.

`olist_analysis` keeps the raw data untouched, so the script can be re-run at any time.

## 4. Next step

With the clean database in place, connect from Python to `olist_clean` for feature engineering and modelling:

- connection: localhost / 3306 / user / password / database `olist_clean`
- interactions: `clean_orders` + `clean_order_items` + `clean_reviews`
- user features: `clean_customers` plus order aggregates
- item features: `clean_products` plus sales and review aggregates
- context features: time and state from `clean_orders` / `clean_customers`, plus delivery timing

The `recsys/` folder implements this step: see `recsys/README.md`.

## 5. Troubleshooting

- "Unknown database" while importing: the `olist_analysis` database was not created first.
- Forgotten root password: reset it through the Navicat connection, or follow the standard MySQL 8
  password-reset procedure.
- The import takes a while: the 7 dumps total about 45 MB, just let it finish.
