"""
Tests for building dj metric queries
"""


import pytest
from sqlmodel import Session

from dj.construction.dj_query import build_dj_metric_query

from ..sql.utils import compare_query_strings


@pytest.mark.asyncio
async def test_build_dj_metric_query(request):
    """
    Test building a metric query
    """
    construction_session: Session = request.getfixturevalue("construction_session")
    query = """
    SELECT basic.num_users_us
    FROM metrics
    GROUP BY basic.dimension.users.country
    """
    expected = """
SELECT  basic_DOT_num_users_us.sum_users
 FROM (SELECT  SUM(basic_DOT_transform_DOT_country_agg.num_users) AS sum_users,
        basic_DOT_transform_DOT_country_agg.country AS a_DOT_country,
        basic_DOT_transform_DOT_country_agg.num_users AS a_DOT_num_users,
        basic_DOT_source_DOT_users.id AS b_DOT_id,
        basic_DOT_source_DOT_users.full_name AS b_DOT_full_name,
        basic_DOT_source_DOT_users.names_map AS b_DOT_names_map,
        basic_DOT_source_DOT_users.user_metadata AS b_DOT_user_metadata,
        basic_DOT_source_DOT_users.age AS b_DOT_age,
        basic_DOT_source_DOT_users.country AS b_DOT_country,
        basic_DOT_source_DOT_users.gender AS b_DOT_gender,
        basic_DOT_source_DOT_users.preferred_language AS b_DOT_preferred_language,
        basic_DOT_source_DOT_users.secret_number AS b_DOT_secret_number
 FROM (SELECT  basic_DOT_source_DOT_users.country,
        COUNT(DISTINCT basic_DOT_source_DOT_users.id) AS num_users
 FROM "basic.source.users" AS basic_DOT_source_DOT_users

 GROUP BY  basic_DOT_source_DOT_users.country) AS basic_DOT_transform_DOT_country_agg
INNER JOIN "basic.source.users" AS basic_DOT_source_DOT_users
        ON basic_DOT_transform_DOT_country_agg.country = basic_DOT_source_DOT_users.country
 WHERE  basic_DOT_transform_DOT_country_agg.country = 'US'
) AS basic_DOT_num_users_us
LEFT JOIN (SELECT  basic_DOT_source_DOT_users.country,
        COUNT(DISTINCT basic_DOT_source_DOT_users.id) AS num_users
 FROM "basic.source.users" AS basic_DOT_source_DOT_users

 GROUP BY  basic_DOT_source_DOT_users.country) AS basic_DOT_transform_DOT_country_agg
        ON basic_DOT_num_users_us.a_DOT_country = basic_DOT_transform_DOT_country_agg.country AND basic_DOT_num_users_us.a_DOT_num_users = basic_DOT_transform_DOT_country_agg.num_users
LEFT JOIN "basic.source.users" AS basic_DOT_source_DOT_users
        ON basic_DOT_num_users_us.b_DOT_id = basic_DOT_source_DOT_users.id AND basic_DOT_num_users_us.b_DOT_full_name = basic_DOT_source_DOT_users.full_name AND basic_DOT_num_users_us.b_DOT_names_map = basic_DOT_source_DOT_users.names_map AND basic_DOT_num_users_us.b_DOT_user_metadata = basic_DOT_source_DOT_users.user_metadata AND basic_DOT_num_users_us.b_DOT_age = basic_DOT_source_DOT_users.age AND basic_DOT_num_users_us.b_DOT_country = basic_DOT_source_DOT_users.country AND basic_DOT_num_users_us.b_DOT_gender = basic_DOT_source_DOT_users.gender AND basic_DOT_num_users_us.b_DOT_preferred_language = basic_DOT_source_DOT_users.preferred_language AND basic_DOT_num_users_us.b_DOT_secret_number = basic_DOT_source_DOT_users.secret_number
LEFT JOIN (SELECT  basic_DOT_source_DOT_users.id,
        basic_DOT_source_DOT_users.full_name,
        basic_DOT_source_DOT_users.age,
        basic_DOT_source_DOT_users.country,
        basic_DOT_source_DOT_users.gender,
        basic_DOT_source_DOT_users.preferred_language,
        basic_DOT_source_DOT_users.secret_number
 FROM "basic.source.users" AS basic_DOT_source_DOT_users

) AS basic_DOT_dimension_DOT_users
        ON basic_DOT_transform_DOT_country_agg.country = basic_DOT_dimension_DOT_users.country AND basic_DOT_transform_DOT_country_agg.country = basic_DOT_dimension_DOT_users.country
 GROUP BY  basic_DOT_dimension_DOT_users.country
    """
    query_ast = build_dj_metric_query(construction_session, query)
    assert compare_query_strings(expected, str(query_ast))
