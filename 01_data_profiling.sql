-- ============================================================
--data profiling
-- Only profile repeatable records, missing values, and dirty data across all tables, providing a basis for following data-cleaning step.
-- ============================================================
USE olist_analysis;
SET NAMES utf8mb4;

--check row counts and primary-key uniqueness for all tables 
SELECT 'orders' AS tbl, COUNT(*) AS rows_total, COUNT(DISTINCT order_id) AS distinct_pk FROM orders
UNION ALL SELECT 'customers', COUNT(*), COUNT(DISTINCT customer_id) FROM customers
UNION ALL SELECT 'order_items', COUNT(*), COUNT(DISTINCT CONCAT(order_id,'|',order_item_id)) FROM order_items
UNION ALL SELECT 'payments', COUNT(*), COUNT(DISTINCT CONCAT(order_id,'|',payment_sequential)) FROM payments
UNION ALL SELECT 'reviews', COUNT(*), COUNT(DISTINCT CONCAT(review_id,'|',order_id)) FROM reviews
UNION ALL SELECT 'products', COUNT(*), COUNT(DISTINCT product_id) FROM products
UNION ALL SELECT 'sellers', COUNT(*), COUNT(DISTINCT seller_id) FROM sellers;

-- 2. check repeated data
-- 2.1 orders：order_id repeated
SELECT COUNT(*) AS dup_orders_rows FROM (
  SELECT order_id FROM orders GROUP BY order_id HAVING COUNT(*) > 1
) t;
-- 2.2 customers：customer_unique_id (one user have many accounts)
SELECT COUNT(*) AS dup_unique_ids FROM (
  SELECT customer_unique_id FROM customers GROUP BY customer_unique_id HAVING COUNT(*) > 1
) t;
-- 2.3 order_items：find the same order has more than one records
SELECT COUNT(*) AS dup_item_groups FROM (
  SELECT order_id, product_id, seller_id FROM order_items
  GROUP BY order_id, product_id, seller_id HAVING COUNT(*) > 1
) t;
-- 2.4 reviews：one review appears several times
SELECT COUNT(*) AS dup_review_ids FROM (
  SELECT review_id FROM reviews GROUP BY review_id HAVING COUNT(*) > 1
) t;
-- 2.5 payments：(order_id, payment_sequential) repeated payment
SELECT COUNT(*) AS dup_pay_rows FROM (
  SELECT order_id, payment_sequential FROM payments
  GROUP BY order_id, payment_sequential HAVING COUNT(*) > 1
) t;
-- 2.6 products / sellers (primary key repeats)
SELECT COUNT(*) AS dup_products FROM (SELECT product_id FROM products GROUP BY product_id HAVING COUNT(*) > 1) t;
SELECT COUNT(*) AS dup_sellers FROM (SELECT seller_id FROM sellers GROUP BY seller_id HAVING COUNT(*) > 1) t;

-- 3. check the missing data
-- 3.1 orders 
SELECT
  SUM(customer_id IS NULL OR customer_id = '') AS null_customer_id,
  SUM(order_purchase_timestamp IS NULL) AS null_purchase_ts,
  SUM(order_status IS NULL OR order_status = '') AS null_status
FROM orders;
-- 3.2 order_items 
SELECT
  SUM(product_id IS NULL OR product_id = '') AS null_product,
  SUM(price IS NULL) AS null_price,
  SUM(freight_value IS NULL) AS null_freight
FROM order_items;
-- 3.3 reviews 
SELECT
  SUM(review_score IS NULL) AS null_score,
  SUM(review_comment_message IS NULL OR review_comment_message = '') AS empty_comment,
  SUM(review_comment_title IS NULL OR review_comment_title = '') AS empty_title
FROM reviews;
-- 3.4 products 
SELECT
  SUM(product_category_name IS NULL OR product_category_name = '') AS null_category,
  SUM(product_weight_g IS NULL) AS null_weight,
  SUM(product_description_length IS NULL) AS null_desc
FROM products;
-- 3.5 payments 
SELECT
  SUM(payment_type IS NULL) AS null_type,
  SUM(payment_value IS NULL) AS null_value
FROM payments;

-- 4. check the dirty data
-- 4.1 check order state
SELECT order_status, COUNT(*) AS cnt FROM orders GROUP BY order_status ORDER BY cnt DESC;
-- 4.2 Timestamp ordering violations: approved < purchased, delivered < shipped, delivered < approved.
SELECT COUNT(*) AS bad_time_orders FROM orders
WHERE order_approved_at < order_purchase_timestamp
   OR (order_delivered_customer_date IS NOT NULL AND order_delivered_carrier_date IS NOT NULL
       AND order_delivered_customer_date < order_delivered_carrier_date)
   OR (order_delivered_customer_date IS NOT NULL AND order_approved_at IS NOT NULL
       AND order_delivered_customer_date < order_approved_at);
-- 4.3 purchasing time not in（2016-09 ~ 2018-10）
SELECT COUNT(*) AS out_of_range FROM orders
WHERE order_purchase_timestamp < '2016-09-01' OR order_purchase_timestamp >= '2018-11-01';
-- 4.4 price < 0
SELECT COUNT(*) AS neg_price FROM order_items WHERE price < 0 OR freight_value < 0;
-- 4.5 rating out of 1-5
SELECT COUNT(*) AS bad_score FROM reviews WHERE review_score NOT BETWEEN 1 AND 5;
-- 4.6 delivery time longer than 30 days but not belong to dirty data
SELECT COUNT(*) AS long_delivery FROM orders
WHERE order_delivered_customer_date IS NOT NULL
  AND TIMESTAMPDIFF(DAY, order_purchase_timestamp, order_delivered_customer_date) > 30;

-- 5. orders with no matching rows
SELECT COUNT(DISTINCT o.order_id) AS orders_without_items
FROM orders o LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE oi.order_id IS NULL;
SELECT COUNT(DISTINCT o.order_id) AS orders_without_payment
FROM orders o LEFT JOIN payments p ON o.order_id = p.order_id
WHERE p.order_id IS NULL;
SELECT COUNT(DISTINCT o.order_id) AS orders_without_review
FROM orders o LEFT JOIN reviews r ON o.order_id = r.order_id
WHERE r.order_id IS NULL;
SELECT COUNT(*) AS orphan_items
FROM order_items oi LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL;

-- 6. Interaction-level profiling: scale, sparsity, and repeat-purchase rate.

--find the  interactions of all the tables
SELECT COUNT(DISTINCT c.customer_unique_id) AS users,
       COUNT(DISTINCT oi.product_id) AS products,
       COUNT(*) AS interactions
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN customers c ON o.customer_id = c.customer_id;

--find the repeat consumption
SELECT COUNT(*) AS total_users,
       SUM(order_cnt >= 2) AS repeat_buyers,
       ROUND(SUM(order_cnt >= 2) * 100.0 / COUNT(*), 2) AS repeat_rate_pct
FROM (
  SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS order_cnt
  FROM orders o JOIN customers c ON o.customer_id = c.customer_id
  GROUP BY c.customer_unique_id
) t;
--items number in each order
SELECT review_score, COUNT(*) AS cnt FROM reviews GROUP BY review_score ORDER BY review_score;
SELECT items_per_order, COUNT(*) AS orders_cnt
FROM (
  SELECT order_id, COUNT(*) AS items_per_order FROM order_items GROUP BY order_id
) t
GROUP BY items_per_order ORDER BY items_per_order;

