# coding: utf-8

"""


    Generated by: https://openapi-generator.tech
"""

import unittest
from unittest.mock import patch

import urllib3

import djclient
from djclient.paths.graphql import post  # noqa: E501
from djclient import configuration, schemas, api_client

from .. import ApiTestMixin


class TestGraphql(ApiTestMixin, unittest.TestCase):
    """
    Graphql unit test stubs
        Handle Http Post  # noqa: E501
    """
    _configuration = configuration.Configuration()

    def setUp(self):
        used_api_client = api_client.ApiClient(configuration=self._configuration)
        self.api = post.ApiForpost(api_client=used_api_client)  # noqa: E501

    def tearDown(self):
        pass

    response_status = 200




if __name__ == '__main__':
    unittest.main()
