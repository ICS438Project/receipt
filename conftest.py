import pytest

def pytest_addoption(parser):
    parser.addoption("--apikey", action="store", help="API key for testing")

@pytest.fixture
def apikey(request):
    return request.config.getoption("--apikey")
