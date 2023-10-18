from typing import Optional

from qiskit_ibm_provider import IBMAccountError, IBMProvider
from qiskit_ibm_provider.api.exceptions import RequestsApiError


def get_provider(
    *, api_token: Optional[str], hub: str, group: str, project: str
) -> IBMProvider:
    """Helper for handling errors from getting an IBMProvider"""
    if api_token is not None and api_token != "None":
        ibm_token = api_token
    else:
        ibm_token = None

    try:
        provider = IBMProvider(token=ibm_token, instance=f"{hub}/{group}/{project}")
    except IBMAccountError as e:
        if api_token is None:
            raise RuntimeError("No providers were found. Missing IBMQ API token?")
        raise RuntimeError(e) from e
    except RequestsApiError as e:
        if e.status_code == 401:
            raise RuntimeError("Unable to log in to IBMQ. Check your API token.")
        raise RuntimeError(e) from e
    return provider
