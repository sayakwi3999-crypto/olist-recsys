-- ============================================================
-- 02 Cleaning, de-duplication and table creation
-- Output: 8 cleaned tables in the olist_clean database + a cleaning_log table,
--         followed by a before/after comparison report.
-- The source database olist_analysis is never modified, so this script is safe to re-run.
-- ============================================================
CREATE DATABASE IF NOT EXISTS olist_clean DEFAULT CHARACTER SET utf8mb4;
USE olist_clean;
SET NAMES utf8mb4;

DROP TABLE IF EXISTS cleaning_log;
CREATE TABLE cleaning_log (
  id INT AUTO_INCREMENT PRIMARY KEY,
  tbl VARCHAR(50) NOT NULL,
  reason VARCHAR(200) NOT NULL,
  rows_affected INT NOT NULL DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------- 1. orders ----------
SET @dup_orders := 0;
SET @null_cust := 0;
SET @bad_time := 0;
SET @out_range := 0;
SELECT COUNT(*) INTO @dup_orders FROM (SELECT order_id FROM olist_analysis.orders GROUP BY order_id HAVING COUNT(*) > 1) t;
SELECT COUNT(*) INTO @null_cust FROM olist_analysis.orders WHERE customer_id IS NULL OR customer_id = '';
SELECT COUNT(*) INTO @bad_time FROM olist_analysis.orders
WHERE order_approved_at < order_purchase_timestamp
   OR (order_delivered_customer_date IS NOT NULL AND order_delivered_carrier_date IS NOT NULL
       AND order_delivered_customer_date < order_delivered_carrier_date)
   OR (order_delivered_customer_date IS NOT NULL AND order_approved_at IS NOT NULL
       AND order_delivered_customer_date < order_approved_at);
SELECT COUNT(*) INTO @out_range FROM olist_analysis.orders
WHERE order_purchase_timestamp < '2016-09-01' OR order_purchase_timestamp >= '2018-11-01';
INSERT INTO cleaning_log(tbl, reason, rows_affected) VALUES
 ('orders', 'duplicate order_id (earliest kept)', @dup_orders),
 ('orders', 'missing customer_id', @null_cust),
 ('orders', 'impossible timestamp order (approval/shipping/delivery out of sequence)', @bad_time),
 ('orders', 'purchase timestamp outside 2016-09 ~ 2018-10', @out_range);

DROP TABLE IF EXISTS clean_orders;
CREATE TABLE clean_orders AS
SELECT order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at,
       order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date
FROM (
  SELECT o.*,
         ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY order_purchase_timestamp ASC, order_id ASC) AS rn
  FROM olist_analysis.orders o
  WHERE (customer_id IS NOT NULL AND customer_id <> '')
    AND order_purchase_timestamp >= '2016-09-01' AND order_purchase_timestamp < '2018-11-01'
    AND NOT (order_approved_at < order_purchase_timestamp)
    AND NOT (order_delivered_customer_date IS NOT NULL AND order_delivered_carrier_date IS NOT NULL
             AND order_delivered_customer_date < order_delivered_carrier_date)
    AND NOT (order_delivered_customer_date IS NOT NULL AND order_approved_at IS NOT NULL
             AND order_delivered_customer_date < order_approved_at)
) o
WHERE rn = 1;
ALTER TABLE clean_orders ADD PRIMARY KEY (order_id);
ALTER TABLE clean_orders ADD INDEX ix_orders_customer (customer_id);

-- ---------- 2. customers ----------
SET @dup_uid := 0;
SET @no_order_cust := 0;
SELECT COUNT(*) INTO @dup_uid FROM (SELECT customer_unique_id FROM olist_analysis.customers GROUP BY customer_unique_id HAVING COUNT(*) > 1) t;
SELECT COUNT(*) INTO @no_order_cust FROM olist_analysis.customers c
LEFT JOIN olist_clean.clean_orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
INSERT INTO cleaning_log(tbl, reason, rows_affected) VALUES
 ('customers', 'duplicate customer_unique_id (multiple accounts, latest kept)', @dup_uid),
 ('customers', 'customer rows without a valid order (dropped with the order)', @no_order_cust);

DROP TABLE IF EXISTS clean_customers;
CREATE TABLE clean_customers AS
SELECT c.customer_id, c.customer_unique_id, c.customer_zip_code_prefix, c.customer_city, c.customer_state
FROM olist_analysis.customers c
JOIN (
  SELECT o.customer_id, c.customer_unique_id,
         ROW_NUMBER() OVER (PARTITION BY c.customer_unique_id ORDER BY o.order_purchase_timestamp DESC, o.order_id DESC) AS rn
  FROM olist_clean.clean_orders o
  JOIN olist_analysis.customers c ON o.customer_id = c.customer_id
  WHERE c.customer_unique_id IS NOT NULL
) o ON o.customer_id = c.customer_id AND o.rn = 1;
ALTER TABLE clean_customers ADD PRIMARY KEY (customer_id);
ALTER TABLE clean_customers ADD INDEX ix_customers_uid (customer_unique_id);

-- ---------- 3. sellers ----------
SET @dup_seller := 0;
SELECT COUNT(*) INTO @dup_seller FROM (SELECT seller_id FROM olist_analysis.sellers GROUP BY seller_id HAVING COUNT(*) > 1) t;
INSERT INTO cleaning_log(tbl, reason, rows_affected) VALUES ('sellers', 'duplicate seller_id', @dup_seller);

DROP TABLE IF EXISTS clean_sellers;
CREATE TABLE clean_sellers AS
SELECT DISTINCT seller_id, seller_zip_code_prefix, seller_city, seller_state FROM olist_analysis.sellers;
ALTER TABLE clean_sellers ADD PRIMARY KEY (seller_id);

-- ---------- 4. products ----------
SET @dup_prod := 0;
SET @null_cat := 0;
SELECT COUNT(*) INTO @dup_prod FROM (SELECT product_id FROM olist_analysis.products GROUP BY product_id HAVING COUNT(*) > 1) t;
SELECT COUNT(*) INTO @null_cat FROM olist_analysis.products WHERE product_category_name IS NULL OR product_category_name = '';
INSERT INTO cleaning_log(tbl, reason, rows_affected) VALUES
 ('products', 'duplicate product_id', @dup_prod),
 ('products', 'missing category (kept, flagged as "uncategorised")', @null_cat);

DROP TABLE IF EXISTS clean_products;
CREATE TABLE clean_products AS
SELECT product_id,
       CASE WHEN product_category_name IS NULL OR product_category_name = '' THEN 'uncategorised'
            ELSE product_category_name END AS product_category_name,
       product_name_length, product_description_length, product_photos_qty,
       product_weight_g, product_length_cm, product_height_cm, product_width_cm
FROM olist_analysis.products;
ALTER TABLE clean_products ADD PRIMARY KEY (product_id);
ALTER TABLE clean_products ADD INDEX ix_products_cat (product_category_name);

-- ---------- 5. order_items ----------
SET @oi_no_order := 0;
SET @oi_neg := 0;
SET @oi_dup_group := 0;
SET @oi_null_prod := 0;
SELECT COUNT(*) INTO @oi_no_order FROM olist_analysis.order_items oi
LEFT JOIN olist_clean.clean_orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL;
SELECT COUNT(*) INTO @oi_neg FROM olist_analysis.order_items WHERE price < 0 OR freight_value < 0;
SELECT COUNT(*) INTO @oi_dup_group FROM (
  SELECT order_id, product_id, seller_id FROM olist_analysis.order_items
  GROUP BY order_id, product_id, seller_id HAVING COUNT(*) > 1
) t;
SELECT COUNT(*) INTO @oi_null_prod FROM olist_analysis.order_items WHERE product_id IS NULL OR product_id = '';
INSERT INTO cleaning_log(tbl, reason, rows_affected) VALUES
 ('order_items', 'order was dropped (cleaned with the order)', @oi_no_order),
 ('order_items', 'negative price', @oi_neg),
 ('order_items', 'same order/item/seller on multiple rows (merged into quantity)', @oi_dup_group),
 ('order_items', 'missing product_id', @oi_null_prod);

DROP TABLE IF EXISTS clean_order_items;
CREATE TABLE clean_order_items AS
SELECT order_id, product_id, seller_id,
       COUNT(*) AS quantity,
       SUM(price) AS price_total,
       SUM(freight_value) AS freight_total
FROM olist_analysis.order_items oi
WHERE oi.order_id IN (SELECT order_id FROM olist_clean.clean_orders)
  AND oi.product_id IS NOT NULL AND oi.product_id <> ''
  AND oi.price >= 0 AND oi.freight_value >= 0
GROUP BY order_id, product_id, seller_id;
ALTER TABLE clean_order_items ADD PRIMARY KEY (order_id, product_id, seller_id);
ALTER TABLE clean_order_items ADD INDEX ix_oi_product (product_id);
ALTER TABLE clean_order_items ADD INDEX ix_oi_seller (seller_id);

-- ---------- 6. payments ----------
SET @pay_no_order := 0;
SET @pay_null := 0;
SELECT COUNT(*) INTO @pay_no_order FROM olist_analysis.payments p
LEFT JOIN olist_clean.clean_orders o ON p.order_id = o.order_id
WHERE o.order_id IS NULL;
SELECT COUNT(*) INTO @pay_null FROM olist_analysis.payments WHERE payment_value IS NULL OR payment_type IS NULL;
INSERT INTO cleaning_log(tbl, reason, rows_affected) VALUES
 ('payments', 'order was dropped (cleaned with the order)', @pay_no_order),
 ('payments', 'missing payment value or type', @pay_null);

DROP TABLE IF EXISTS clean_payments;
CREATE TABLE clean_payments AS
SELECT order_id, payment_sequential, payment_type, payment_installments, payment_value
FROM olist_analysis.payments p
WHERE p.order_id IN (SELECT order_id FROM olist_clean.clean_orders)
  AND payment_value IS NOT NULL AND payment_type IS NOT NULL;
ALTER TABLE clean_payments ADD PRIMARY KEY (order_id, payment_sequential);

-- Order-level payment summary (used later for GMV / average order value analysis)
DROP TABLE IF EXISTS order_payment_summary;
CREATE TABLE order_payment_summary AS
SELECT order_id,
       COUNT(*) AS payment_count,
       SUM(payment_value) AS total_payment_value,
       MAX(payment_installments) AS max_installments,
       GROUP_CONCAT(DISTINCT payment_type ORDER BY payment_type SEPARATOR ',') AS payment_types
FROM olist_clean.clean_payments
GROUP BY order_id;
ALTER TABLE order_payment_summary ADD PRIMARY KEY (order_id);

-- ---------- 7. reviews ----------
SET @rv_no_order := 0;
SET @rv_bad_score := 0;
SET @rv_dup := 0;
SET @rv_empty_comment := 0;
SELECT COUNT(*) INTO @rv_no_order FROM olist_analysis.reviews r
LEFT JOIN olist_clean.clean_orders o ON r.order_id = o.order_id
WHERE o.order_id IS NULL;
SELECT COUNT(*) INTO @rv_bad_score FROM olist_analysis.reviews WHERE review_score NOT BETWEEN 1 AND 5;
SELECT COUNT(*) INTO @rv_dup FROM (SELECT review_id FROM olist_analysis.reviews GROUP BY review_id HAVING COUNT(*) > 1) t;
SELECT COUNT(*) INTO @rv_empty_comment FROM olist_analysis.reviews WHERE review_comment_message IS NULL OR review_comment_message = '';
INSERT INTO cleaning_log(tbl, reason, rows_affected) VALUES
 ('reviews', 'order was dropped (cleaned with the order)', @rv_no_order),
 ('reviews', 'review score outside the 1-5 range', @rv_bad_score),
 ('reviews', 'review_id reused across orders (kept, flagged as shared review)', @rv_dup),
 ('reviews', 'missing review text (kept, needs handling in sentiment analysis)', @rv_empty_comment);

DROP TABLE IF EXISTS clean_reviews;
CREATE TABLE clean_reviews AS
SELECT r.review_id, r.order_id, r.review_score, r.review_comment_title, r.review_comment_message,
       r.review_creation_date, r.review_answer_timestamp
FROM olist_analysis.reviews r
WHERE r.order_id IN (SELECT order_id FROM olist_clean.clean_orders)
  AND r.review_score BETWEEN 1 AND 5;
ALTER TABLE clean_reviews ADD PRIMARY KEY (review_id, order_id);
ALTER TABLE clean_reviews ADD INDEX ix_reviews_order (order_id);

-- ---------- 8. before/after comparison report ----------
SELECT 'orders' AS tbl,
       (SELECT COUNT(*) FROM olist_analysis.orders) AS raw_rows,
       (SELECT COUNT(*) FROM olist_clean.clean_orders) AS clean_rows,
       (SELECT COUNT(*) FROM olist_analysis.orders) - (SELECT COUNT(*) FROM olist_clean.clean_orders) AS removed_rows
UNION ALL
SELECT 'customers',
       (SELECT COUNT(*) FROM olist_analysis.customers),
       (SELECT COUNT(*) FROM olist_clean.clean_customers),
       (SELECT COUNT(*) FROM olist_analysis.customers) - (SELECT COUNT(*) FROM olist_clean.clean_customers)
UNION ALL
SELECT 'order_items (after merge)',
       (SELECT COUNT(*) FROM olist_analysis.order_items),
       (SELECT COUNT(*) FROM olist_clean.clean_order_items),
       (SELECT COUNT(*) FROM olist_analysis.order_items) - (SELECT COUNT(*) FROM olist_clean.clean_order_items)
UNION ALL
SELECT 'payments',
       (SELECT COUNT(*) FROM olist_analysis.payments),
       (SELECT COUNT(*) FROM olist_clean.clean_payments),
       (SELECT COUNT(*) FROM olist_analysis.payments) - (SELECT COUNT(*) FROM olist_clean.clean_payments)
UNION ALL
SELECT 'reviews',
       (SELECT COUNT(*) FROM olist_analysis.reviews),
       (SELECT COUNT(*) FROM olist_clean.clean_reviews),
       (SELECT COUNT(*) FROM olist_analysis.reviews) - (SELECT COUNT(*) FROM olist_clean.clean_reviews)
UNION ALL
SELECT 'products',
       (SELECT COUNT(*) FROM olist_analysis.products),
       (SELECT COUNT(*) FROM olist_clean.clean_products),
       (SELECT COUNT(*) FROM olist_analysis.products) - (SELECT COUNT(*) FROM olist_clean.clean_products)
UNION ALL
SELECT 'sellers',
       (SELECT COUNT(*) FROM olist_analysis.sellers),
       (SELECT COUNT(*) FROM olist_clean.clean_sellers),
       (SELECT COUNT(*) FROM olist_analysis.sellers) - (SELECT COUNT(*) FROM olist_clean.clean_sellers);

-- Detail of every cleaning reason
SELECT tbl, reason, rows_affected FROM olist_clean.cleaning_log ORDER BY id;
