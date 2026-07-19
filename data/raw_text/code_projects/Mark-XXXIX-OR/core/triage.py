import requests
import logging
import config

logger = logging.getLogger("core.triage")

class TriageManager:
    def __init__(self, endpoint: str = "http://127.0.0.1:1234/v1/chat/completions",
                 timeout: float = None):
        self.endpoint = endpoint
        self.timeout = timeout if timeout is not None else config.TRIAGE_TIMEOUT
        self._cached_model = None
        self.system_prompt = (
            'Analyze the following user request. '
            'If it is a simple greeting, routine info, or small task, respond with "[LOCAL]". '
            'If it requires complex planning, deep research, or tool orchestration, respond with "[CLOUD]". '
            'Response must be only the tag.'
        )

    def route_query(self, user_query: str) -> str:
        """
        Categorize the query into [LOCAL] or [CLOUD].
        If LM Studio is offline, times out, or returns an error, fallback to [CLOUD].
        """
        if not user_query:
            return "[CLOUD]"

        # Auto-detect loaded model name from LM Studio
        try:
            models_url = self.endpoint.rsplit("/", 1)[0] + "/models"
            resp = requests.get(models_url, timeout=1.0)
            if resp.status_code == 200:
                models_data = resp.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    model_name = models_data["data"][0]["id"]
                    logger.info(f"TriageManager: Loaded model detected: '{model_name}'")
                else:
                    logger.warning("TriageManager: No model loaded in LM Studio. Routing to [CLOUD].")
                    return "[CLOUD]"
            else:
                logger.warning(f"TriageManager: LM Studio models endpoint returned status {resp.status_code}. Routing to [CLOUD].")
                return "[CLOUD]"
        except Exception as e:
            logger.warning(f"TriageManager: LM Studio is offline or unreachable: {e}. Routing to [CLOUD].")
            return "[CLOUD]"

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.0,
            "max_tokens": 10
        }

        try:
            logger.info(f"TriageManager: routing query to LM Studio at {self.endpoint}...")
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                logger.info(f"TriageManager: LM Studio response: '{content}'")

                # Check for tag
                if "[LOCAL]" in content:
                    return "[LOCAL]"
                elif "[CLOUD]" in content:
                    return "[CLOUD]"

                # If the tag is not exactly [LOCAL] or [CLOUD] but contains the words
                if "LOCAL" in content.upper():
                    return "[LOCAL]"

            else:
                logger.warning(f"TriageManager: LM Studio returned status code {response.status_code}")

        except requests.exceptions.Timeout:
            logger.warning(f"TriageManager: Connection to LM Studio timed out ({self.timeout}s limit). Routing to [CLOUD].")
        except requests.exceptions.ConnectionError:
            logger.warning("TriageManager: LM Studio is offline or connection refused. Routing to [CLOUD].")
        except Exception as e:
            logger.error(f"TriageManager: Error during local triage: {e}. Routing to [CLOUD].")

        return "[CLOUD]"
