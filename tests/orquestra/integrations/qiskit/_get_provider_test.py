from unittest.mock import Mock, create_autospec

import pytest
from qiskit_ibm_provider import IBMAccountError
from qiskit_ibm_provider.api.exceptions import RequestsApiError

from orquestra.integrations.qiskit import _get_provider


@pytest.fixture
def hub():
    return "ibm-q"


@pytest.fixture
def group():
    return "open"


@pytest.fixture
def project():
    return "main"


@pytest.fixture
def mock_ibmprovider(monkeypatch: pytest.MonkeyPatch):
    provider = create_autospec(_get_provider.IBMProvider)
    monkeypatch.setattr(
        _get_provider,
        "IBMProvider",
        provider,
    )
    return provider


@pytest.mark.parametrize(
    "api_token, expected_token",
    (
        (None, None),
        ("None", None),
        ("", ""),
        ("token", "token"),
    ),
)
def test_check_args_handling(
    mock_ibmprovider: Mock,
    hub: str,
    group: str,
    project: str,
    api_token: str,
    expected_token: str,
):
    _ = _get_provider.get_provider(
        api_token=api_token, hub=hub, group=group, project=project
    )

    mock_ibmprovider.assert_called_with(
        token=expected_token, instance=f"{hub}/{group}/{project}"
    )


def test_raises_runtime_error_on_empty_api_token(hub: str, group: str, project: str):
    api_token = ""
    with pytest.raises(RuntimeError):
        _ = _get_provider.get_provider(
            api_token=api_token, hub=hub, group=group, project=project
        )


def test_raises_runtime_error_on_account_error_none_token(
    mock_ibmprovider: Mock, hub: str, group: str, project: str
):
    api_token = None
    mock_ibmprovider.side_effect = IBMAccountError("account problem")
    with pytest.raises(RuntimeError) as exc_info:
        _ = _get_provider.get_provider(
            api_token=api_token, hub=hub, group=group, project=project
        )
    exc_info.match("No providers were found. Missing IBMQ API token?")


def test_raises_runtime_error_on_unauthorized_token(
    mock_ibmprovider: Mock, hub: str, group: str, project: str
):
    api_token = "bad token"
    mock_ibmprovider.side_effect = RequestsApiError("login error", 401)
    with pytest.raises(RuntimeError) as exc_info:
        _ = _get_provider.get_provider(
            api_token=api_token, hub=hub, group=group, project=project
        )
    exc_info.match("Unable to log in to IBMQ. Check your API token.")


def test_raises_runtime_error_on_unknown_api_error(
    mock_ibmprovider: Mock, hub: str, group: str, project: str
):
    api_token = "bad token"
    mock_ibmprovider.side_effect = RequestsApiError("Unknown error")
    with pytest.raises(RuntimeError) as exc_info:
        _ = _get_provider.get_provider(
            api_token=api_token, hub=hub, group=group, project=project
        )
    exc_info.match("Unknown error")
