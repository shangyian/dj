"""Tests the celery app."""
import datetime
import json
from dataclasses import dataclass
from unittest.mock import call

import pytest
from urllib3._collections import HTTPHeaderDict

from djrs.worker.tasks import reflect, refresh


def test_refresh(celery_app, mocker):
    """
    Tests that the reflection service refreshes DJ nodes' availability states.
    """

    @dataclass
    class MockResponse:
        """
        Mocking responses to uri requests
        """

        data: str
        status: int

    def mock_requests(*args, **kwargs):  # pylint: disable=unused-argument
        """
        Mock out the different DJ API requests
        """
        if "http://dj:8000/catalogs/" in args:
            return MockResponse(
                data="""
            [
                {
                    "name": "postgres",
                    "engines": [
                    {
                        "name": "postgres",
                        "version":"",
                        "uri":""
                    }]
                }
            ]
            """,
                status=200,
            )
        return MockResponse(
            data="""
        [
            {
                "name": "random",
                "catalog": {"name": "postgres"},
                "schema_": "test",
                "table": "revenue"
            },
            {
                "name": "random2",
                "catalog": {"name": "postgres"},
                "schema_": null,
                "table": null
            }
        ]
        """,
            status=200,
        )

    mocker.patch("urllib3.PoolManager.request", mock_requests)
    refresh.apply(args=()).get()

    assert {"djrs.worker.app.refresh", "djrs.worker.tasks.reflect"}.intersection(
        celery_app.tasks.keys(),
    )


def test_refresh_failure(celery_app, mocker):  # pylint: disable=unused-argument
    """
    Tests that refresh task failures are handled.
    """
    mock_requests = mocker.patch("urllib3.PoolManager.request")
    mock_requests.return_value.data = """
    [
        {
            "name": "random",
            "catalog": null,
            "schema_": null,
            "table": null,
        }
    ]
    """
    mock_requests.return_value.status = 200
    with pytest.raises(json.decoder.JSONDecodeError):
        refresh.apply(args=()).get()


def test_reflect(celery_app, mocker, freezer):  # pylint: disable=unused-argument
    """
    Tests the reflection task.
    """
    now = datetime.datetime.now()
    mock_djqs_requests = mocker.patch("requests.get")
    mock_djqs_requests.return_value.json = lambda: {
        "name": "postgres.test.revenue",
        "columns": [{"name": "id", "type": "int"}],
    }
    mock_djqs_requests.return_value.status_code = 200

    mock_dj_requests = mocker.patch("urllib3.PoolManager.request")
    mock_dj_requests.return_value.data = """
    [
        {
            "name": "random",
            "catalog": {"name": "postgres"},
            "schema_": "test",
            "table": "revenue",
            "engines": [{"name":"spark","version":"3.3.2","uri":"spark://local[*]"}]
        }
    ]
    """
    mock_dj_requests.return_value.status = 200

    reflect.apply(
        args=(
            "random",
            "postgres",
            "test",
            "revenue",
            {"name": "spark", "version": "3.3.2", "uri": "spark://local[*]"},
        ),
    ).get()

    mock_djqs_requests.assert_called_once_with(
        "http://djqs:8001/table/postgres.test.revenue/columns/",
        params={"engine": "spark", "engine_version": "3.3.2"},
        timeout=30,
    )
    assert mock_dj_requests.call_args_list == [
        call(
            "PATCH",
            "http://dj:8000/nodes/random/",
            body=b'{"columns":{"id":{"type":"int"}}}',
            preload_content=True,
            timeout=None,
            headers=HTTPHeaderDict(
                {
                    "User-Agent": "OpenAPI-Generator/1.0.0/python",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            ),
        ),
        call(
            "POST",
            "http://dj:8000/data/random/availability/",
            body=b'{"catalog":"postgres","max_partition":[],'
            b'"min_partition":[],"table":"revenue",'
            b'"valid_through_ts":'
            + f"{int(now.timestamp())}".encode()
            + b',"schema_":"test"}',
            preload_content=True,
            timeout=None,
            headers=HTTPHeaderDict(
                {
                    "User-Agent": "OpenAPI-Generator/1.0.0/python",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            ),
        ),
    ]
