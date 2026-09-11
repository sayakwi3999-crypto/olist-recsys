# -*- coding: utf-8 -*-
"""Export the cleaned tables from MySQL (olist_clean) to local CSV files."""
import time
from pathlib import Path

import pandas as pd
import pymysql

from config import DB_CONFIG, RAW_DIR

TABLES = {
    "clean_customers": "SELECT customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state FROM clean_customers",
    # Full user mapping: every historical account of one customer_unique_id belongs to the same user.
    # The cleaning step keeps only the latest account, so this mapping must come from the
    # raw customers table - otherwise repeat-purchase history would be lost.
    "user_map": (
        "SELECT DISTINCT c.customer_id, c.customer_unique_id, c.customer_state "
        "FROM olist_analysis.customers c "
        "WHERE c.customer_id IN (SELECT customer_id FROM olist_clean.clean_orders)"
    ),
    "clean_orders": (
        "SELECT order_id, customer_id, order_status, order_purchase_timestamp, "
        "order_delivered_customer_date, order_estimated_delivery_date FROM clean_orders"
    ),
    "clean_order_items": (
        "SELECT order_id, product_id, seller_id, quantity, price_total, freight_total FROM clean_order_items"
    ),
    "clean_reviews": "SELECT review_id, order_id, review_score FROM clean_reviews",
    "clean_products": (
        "SELECT product_id, product_category_name, product_name_length, product_description_length, "
        "product_photos_qty, product_weight_g FROM clean_products"
    ),
    "order_payment_summary": (
        "SELECT order_id, total_payment_value, payment_count, max_installments, payment_types "
        "FROM order_payment_summary"
    ),
}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_CONFIG.get("password"):
        raise SystemExit(
            "Missing database password: copy recsys/.env.example to recsys/.env and fill in "
            "DB_PASSWORD, or set the same environment variable."
        )
    conn = pymysql.connect(**DB_CONFIG)
    try:
        for name, sql in TABLES.items():
            t0 = time.time()
            df = pd.read_sql(sql, conn)
            out = RAW_DIR / f"{name}.csv"
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"[export] {name}: {len(df):>8,} rows -> {out.name} ({time.time()-t0:.1f}s)")
    finally:
        conn.close()
    print("[export] All tables exported.")


if __name__ == "__main__":
    main()
