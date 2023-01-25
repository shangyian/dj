--
-- Basic example
--
CREATE SCHEMA IF NOT EXISTS basic;
CREATE SCHEMA IF NOT EXISTS metrics_repo;

--
-- basic.dim_users
--
CREATE TABLE IF NOT EXISTS basic.dim_users (
  id integer PRIMARY KEY,
  full_name text,
  age integer,
  country text,
  gender text,
  preferred_language text
);

INSERT INTO basic.dim_users (id, full_name, age, country, gender, preferred_language)
  VALUES
    (1, 'Alice One', 10, 'Argentina', 'female', 'Spanish'),
    (2, 'Bob Two', 15, 'Brazil', 'male', 'Portuguese'),
    (3, 'Charlie Three', 20, 'Chile', 'non-binary', 'Spanish'),
    (4, 'Denise Four', 25, 'Denmark', 'female', 'Danish'),
    (5, 'Ernie Five', 27, 'Equator', 'male', 'Spanish'),
    (6, 'Fabian Six', 29, 'France', 'non-binary', 'French')
;

--
-- basic.comments
--
CREATE TABLE IF NOT EXISTS basic.comments (
  id integer PRIMARY KEY,
  user_id integer,
  "timestamp" timestamp with time zone,
  "text" text,
  CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES basic.dim_users (id)
);

INSERT INTO basic.comments (id, user_id, "timestamp", "text")
  VALUES
    (1, 1, '2021-01-01 01:00:00', 'Hola!'),
    (2, 2, '2021-01-01 02:00:00', 'Oi, tudo bom?'),
    (3, 3, '2021-01-01 03:00:00', 'Que pasa?'),
    (4, 4, '2021-01-01 04:00:00', 'Også mig'),
    (5, 5, '2021-01-01 05:00:00', 'Bueno'),
    (6, 6, '2021-01-01 06:00:00', 'Bonjour!'),
    (7, 2, '2021-01-01 07:00:00', 'Prazer em conhecer'),
    (8, 3, '2021-01-01 08:00:00', 'Si, si'),
    (9, 4, '2021-01-01 09:00:00', 'Hej'),
    (10, 5, '2021-01-01 10:00:00', 'Por supuesto'),
    (11, 6, '2021-01-01 11:00:00', 'Oui, oui'),
    (12, 3, '2021-01-01 12:00:00', 'Como no?'),
    (13, 4, '2021-01-01 13:00:00', 'Farvel'),
    (14, 5, '2021-01-01 14:00:00', 'Hola, amigo!'),
    (15, 6, '2021-01-01 15:00:00', 'Très bien'),
    (16, 4, '2021-01-01 16:00:00', 'Dejligt at møde dig'),
    (17, 5, '2021-01-01 17:00:00', 'Dale!'),
    (18, 6, '2021-01-01 18:00:00', 'Bien sûr!'),
    (19, 5, '2021-01-01 19:00:00', 'Hasta luego!'),
    (20, 6, '2021-01-01 20:00:00', 'À toute à l'' heure ! '),
    (21, 6, '2021-01-01 21:00:00', 'Peut être'),
    (22, 6, '2021-01-01 00:00:00', 'Cześć!')
;

CREATE TABLE IF NOT EXISTS metrics_repo.ab_nm_alloc_a(
    test_cell_nbr integer,
    tenure_grouping integer,
    country_iso_code text,
    has_completed_signup integer,
    mop_custom_category_id integer,
    is_p1_possibly_completed integer,
    is_other_exclusion integer,
    is_former_member integer,
    allocation_device_sk integer,
    is_browser integer,
    signup_price_plan_id integer,
    current_price_plan_id integer,
    is_signup_price_plan_migrated integer,
    is_fraud integer,
    is_suspend_resume integer,
    is_bot integer,
    custom_filter_1 text,
    custom_filter_2 text,
    custom_filter_3 text,
    browser_name_category_id integer,
    is_facebook_referrer integer,
    is_free_trial_at_signup integer,
    alloc_group_id text,
    alloc_membership_status integer,
    membership_status integer,
    is_nft_country integer,
    is_mds integer,
    is_fraud_cancelled integer, has_paid_p2 integer, updated_date integer,
    allocation_cnt integer,
    registration_cnt integer,
    provided_mop_cnt integer,
    completed_signup_cnt integer,
    completed_signup_valid_price_plan_cnt integer,
    p1_possibly_completed_cnt integer, p2_possibly_paid_cnt integer, vol_cancel_cnt integer, invol_cancel_cnt integer,
    current_member_cnt integer, retained_cnt integer,
    signup_plan_usd_price_sum DECIMAL,
    signup_plan_usd_price_squared_sum DECIMAL,
    multi_cell_test_cnt integer, gross_realized_revenue_sum DECIMAL, gross_realized_revenue_squared_sum DECIMAL,
    net_realized_revenue_sum DECIMAL, net_realized_revenue_squared_sum DECIMAL,
    days_paid_sum integer,
    days_paid_squared_sum integer, days_with_service_sum integer, days_with_service_squared_sum integer,
    test_id integer, allocation_region_date integer
);

-- COPY metrics_repo.ab_nm_alloc_a(test_cell_nbr, tenure_grouping, country_iso_code, has_completed_signup, mop_custom_category_id, is_p1_possibly_completed, is_other_exclusion, is_former_member, allocation_device_sk, is_browser, signup_price_plan_id, current_price_plan_id, is_signup_price_plan_migrated, is_fraud, is_suspend_resume, is_bot, custom_filter_1, custom_filter_2, custom_filter_3, browser_name_category_id, is_facebook_referrer, is_free_trial_at_signup, alloc_group_id, alloc_membership_status, membership_status, is_nft_country, is_mds, is_fraud_cancelled, has_paid_p2, updated_date, allocation_cnt, registration_cnt, provided_mop_cnt, completed_signup_cnt, completed_signup_valid_price_plan_cnt, p1_possibly_completed_cnt, p2_possibly_paid_cnt, vol_cancel_cnt, invol_cancel_cnt, current_member_cnt, retained_cnt, signup_plan_usd_price_sum, signup_plan_usd_price_squared_sum, multi_cell_test_cnt, gross_realized_revenue_sum, gross_realized_revenue_squared_sum, net_realized_revenue_sum, net_realized_revenue_squared_sum, days_paid_sum, days_paid_squared_sum, days_with_service_sum, days_with_service_squared_sum, test_id, allocation_region_date)
-- FROM 'ab_nm_alloc_a.csv'
-- DELIMITER ','
-- CSV HEADER;


CREATE TABLE IF NOT EXISTS metrics_repo.ab_test_detail_d_v2(
    test_id integer,
    "name" text,
    test_type text
);

INSERT INTO metrics_repo.ab_test_detail_d_v2 (test_id, "name", test_type)
  VALUES
    (49723, 'blah', 'MEMBER'),
    (22222, 'blahblah', 'MEMBER')
;
