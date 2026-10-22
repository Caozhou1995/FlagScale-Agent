# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool base class."""

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict


class ImageResult(str):
    """A tool result that is plain text PLUS attached images for the model to see.

    Subclassing ``str`` (not tuple) is deliberate: every existing consumer of a
    tool result — the registry's ``len()`` truncation, guard post-checks, the
    kernel's result plumbing — keeps working unchanged, because an ImageResult
    IS a string. The images ride along as an attribute and are only inspected by
    the provider when it builds the request body.

    ``images`` is a list of dicts ``{"media_type": "image/png", "data": <b64>}``.
    """

    images: list

    def __new__(cls, text, images):
        obj = super().__new__(cls, text)
        obj.images = list(images or [])
        return obj

    def truncated_copy(self, text):
        """Rebuild with replacement text but the same images (for truncation)."""
        return ImageResult(text, self.images)


class Tool(ABC):
    """Base class for all agent tools.

    Subclasses must set name, description, parameters and implement execute().
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    max_result_size: int = 50000

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        ...

    def _inject_override_param(self, params: dict) -> dict:
        """Inject _override_reason as an optional parameter into schema.

        This allows LLM to bypass guard blocks by providing a reason.
        The field is stripped from tool_args before execute() is called.
        Only injected if the schema has a 'properties' dict.
        """
        if "properties" not in params:
            return params
        params = copy.deepcopy(params)
        params["properties"]["_override_reason"] = {
            "description": (
                "If a previous tool call was blocked by a guard, provide a reason "
                "here to override the block and force execution."
            ),
            "type": "string",
        }
        return params

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._inject_override_param(self.parameters),
            },
        }

    def to_anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._inject_override_param(self.parameters),
        }
