"""AWS Secrets Manager utility for StruktX.

This module provides utilities for fetching and injecting AWS secrets into environment variables
during app creation, before any handlers are built.
"""

from __future__ import annotations

import os
from .logging import get_logger
from typing import Any, Dict, Optional

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class AWSSecretsManager:
    """AWS Secrets Manager client for fetching and injecting secrets."""

    def __init__(
        self,
        region_name: str = "",
        secret_name: str = "",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.region_name = region_name
        self.secret_name = secret_name
        self.access_key = access_key
        self.secret_key = secret_key
        self._secrets_cache: Optional[Dict[str, Any]] = None
        self._log = get_logger(__name__)

        if not BOTO3_AVAILABLE:
            self._log.warn(
                "boto3 not available, secrets will be fetched from environment variables"
            )

    def get_secrets(self) -> Dict[str, Any]:
        """Fetch secrets from AWS Secrets Manager or environment variables."""
        if self._secrets_cache is not None:
            return self._secrets_cache

        if BOTO3_AVAILABLE and self.region_name and self.secret_name:
            try:
                # Create session with optional credentials
                session_kwargs = {}
                if self.access_key and self.secret_key:
                    session_kwargs["aws_access_key_id"] = self.access_key
                    session_kwargs["aws_secret_access_key"] = self.secret_key

                session = boto3.session.Session(**session_kwargs)
                client = session.client(
                    service_name="secretsmanager", region_name=self.region_name
                )

                get_secret_value_response = client.get_secret_value(
                    SecretId=self.secret_name
                )
                secret_string = get_secret_value_response.get("SecretString")

                if not secret_string:
                    raise ValueError(
                        "SecretString is empty in AWS Secrets Manager response."
                    )

                import json

                secrets = json.loads(secret_string)
                self._secrets_cache = secrets
                self._log.info("Successfully fetched secrets from AWS Secrets Manager")
                return secrets

            except ClientError as e:
                self._log.error(
                    f"Failed to fetch secrets from AWS Secrets Manager: {e}"
                )
                raise e
        else:
            raise ValueError(
                "AWS Secrets Manager is not available or missing required configuration"
            )

    def inject_secrets_into_env(self) -> None:
        """Inject all secrets into environment variables."""
        try:
            secrets = self.get_secrets()

            # Inject all secrets into environment variables
            for key, value in secrets.items():
                if isinstance(value, str):
                    os.environ[key] = value
                    self._log.debug(f"Injected secret {key} into environment")

            self._log.info(
                f"Injected {len(secrets)} secrets into environment variables"
            )

        except Exception as e:
            self._log.error(f"Failed to inject secrets into environment: {e}")
            # Continue without secrets if injection fails

    def clear_cache(self) -> None:
        """Clear the secrets cache to force a fresh fetch."""
        self._secrets_cache = None


def inject_aws_secrets(
    region_name: str = "",
    secret_name: str = "",
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> None:
    """Convenience function to inject AWS secrets into environment variables.
    Args:
        region_name: AWS region name
        secret_name: AWS Secrets Manager secret name
        access_key: Optional AWS access key
        secret_key: Optional AWS secret key
    """
    secrets_manager = AWSSecretsManager(
        region_name=region_name,
        secret_name=secret_name,
        access_key=access_key,
        secret_key=secret_key,
    )
    secrets_manager.inject_secrets_into_env()
