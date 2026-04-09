import os
import logging

from openai import AzureOpenAI, OpenAI


logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class GPTClientFactory:
    def __init__(self):
        self.auth_mode = os.getenv("OPENAI_AUTH_MODE", "entra_id").lower()
        self.base_url = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        timeout_value = os.getenv("OPENAI_TIMEOUT", "120")
        self.timeout = float(timeout_value) if timeout_value else 120.0
        self.client = None

    def _init_entra_id_client(self) -> None:
        """Initialize client with Entra ID (Azure AD) authentication."""
        try:
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                ManagedIdentityCredential,
                get_bearer_token_provider,
            )
        except ImportError:
            raise LLMError(
                "azure-identity package is required for Entra ID authentication. "
                "Install with: pip install azure-identity"
            )

        if not self.base_url:
            raise LLMError("AZURE_OPENAI_ENDPOINT or OPENAI_BASE_URL is required for Entra ID authentication")
        if not self.azure_api_version:
            raise LLMError("AZURE_OPENAI_API_VERSION is required for Entra ID authentication")

        credential = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            azure_endpoint=self.base_url,
            azure_ad_token_provider=token_provider,
            api_version=self.azure_api_version,
            timeout=self.timeout,
        )

        logger.debug(f"Initialized Azure OpenAI client with Entra ID authentication (endpoint: {self.base_url})")

    def _init_api_key_client(self) -> None:
        """Initialize client with API Key authentication."""
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is required for APIKey authentication")

        if self.base_url and ("azure" in self.base_url.lower() or "cognitiveservices" in self.base_url.lower()):
            if not self.azure_api_version:
                raise LLMError("AZURE_OPENAI_API_VERSION is required for Azure OpenAI APIKey authentication")
            self.client = AzureOpenAI(
                azure_endpoint=self.base_url,
                api_key=self.api_key,
                api_version=self.azure_api_version,
                timeout=self.timeout,
            )

            logger.debug(f"Initialized Azure OpenAI client with API Key authentication (endpoint: {self.base_url})")
        else:
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)

            logger.debug(f"Initialized OpenAI client with API Key authentication (base_url: {self.base_url})")

    def build(self):
        if self.auth_mode in ["entra", "entra_id", "aad", "azure_ad"]:
            self._init_entra_id_client()
        elif self.auth_mode in ["api_key", "apikey", "key"]:
            self._init_api_key_client()
        else:
            raise LLMError(f"Unsupported OPENAI_AUTH_MODE: {self.auth_mode}. Use 'entra_id' or 'api_key'.")
        return self.client
