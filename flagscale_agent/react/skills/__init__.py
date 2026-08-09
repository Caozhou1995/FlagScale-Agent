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

"""Skill manager — load and parse SKILL.md files."""

import os

from typing import Dict, List, Optional, Tuple

import yaml



class SkillManager:
    """Manages skill loading from prioritized directories."""

    def __init__(self, dirs: List[str]):
        self._dirs = dirs
        self._scan_cache: Optional[Dict[str, str]] = None
        self._list_cache: Optional[List[Dict[str, str]]] = None

    def invalidate_cache(self):
        """Invalidate cached scan/list results. Call after skill dirs change."""
        self._scan_cache = None
        self._list_cache = None

    def _scan(self) -> Dict[str, str]:
        """Build mapping: skill_name -> skill_file_path (later dirs override). Cached."""
        if self._scan_cache is not None:
            return self._scan_cache
        mapping = {}
        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                skill_file = os.path.join(d, entry, "SKILL.md")
                if os.path.isfile(skill_file):
                    try:
                        meta, _ = self._parse_file(skill_file)
                        name = meta.get("name", entry)
                    except Exception:
                        name = entry
                    mapping[name] = skill_file
                    mapping[entry] = skill_file
        self._scan_cache = mapping
        return mapping

    def list_skills(self) -> List[Dict[str, str]]:
        """Scan all directories and return available skills (deduplicated). Cached."""
        if self._list_cache is not None:
            return self._list_cache
        seen_paths = {}
        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                skill_file = os.path.join(d, entry, "SKILL.md")
                if os.path.isfile(skill_file):
                    try:
                        meta, _ = self._parse_file(skill_file)
                        seen_paths[skill_file] = {
                            "name": meta.get("name", entry),
                            "description": meta.get("description", ""),
                            "parameters": meta.get("parameters", []),
                        }
                    except Exception:
                        seen_paths[skill_file] = {"name": entry, "description": "", "parameters": []}
        self._list_cache = list(seen_paths.values())
        return self._list_cache

    def load(self, name: str, **params) -> str:
        """Load a skill by frontmatter name or directory name.

        Optional keyword arguments are substituted into {param_name} placeholders
        in the skill body. Parameters defined in frontmatter with defaults are
        used when not provided by the caller.
        """
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            raise FileNotFoundError(f"Skill '{name}' not found in: {self._dirs}")
        meta, body = self._parse_file(skill_file)
        skill_name = meta.get("name", name)

        # Apply defaults from frontmatter parameters
        param_defs = meta.get("parameters", [])
        if isinstance(param_defs, list):
            for pdef in param_defs:
                if isinstance(pdef, dict):
                    pname = pdef.get("name", "")
                    if pname and pname not in params and "default" in pdef:
                        params[pname] = pdef["default"]

        # Parameter substitution
        for k, v in params.items():
            body = body.replace(f"{{{k}}}", str(v))

        return f"<skill name=\"{skill_name}\">\n{body}\n</skill>"

    def get_meta(self, name: str) -> Dict:
        """Get skill frontmatter metadata without loading full content."""
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            return {}
        try:
            meta, _ = self._parse_file(skill_file)
            return meta
        except Exception:
            return {}

    def _parse_file(self, path: str) -> Tuple[dict, str]:
        """Read a SKILL.md and split YAML frontmatter from body."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._parse_frontmatter(content)

    @staticmethod
    def _parse_frontmatter(content: str) -> Tuple[dict, str]:
        """Split --- delimited YAML frontmatter from markdown body."""
        if not content.startswith("---"):
            return {}, content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content
        try:
            meta = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            meta = {}
        body = parts[2].strip()
        return meta, body
