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
import re

from typing import Dict, List, Optional, Tuple

import yaml



class SkillManager:
    """Manages skill loading from prioritized directories."""

    def __init__(self, dirs: List[str]):
        self._dirs = dirs
        self._scan_cache: Optional[Dict[str, str]] = None
        self._list_cache: Optional[List[Dict[str, str]]] = None
        self._meta_cache: Dict[str, Dict] = {}

    def invalidate_cache(self):
        """Invalidate cached scan/list results. Call after skill dirs change."""
        self._scan_cache = None
        self._list_cache = None
        self._meta_cache = {}

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
                            "keywords": meta.get("keywords", []),
                            "parameters": meta.get("parameters", []),
                            "requires": meta.get("requires", []) or [],
                            "suggests": meta.get("suggests", []) or [],
                        }
                    except Exception:
                        seen_paths[skill_file] = {"name": entry, "description": "", "keywords": [], "parameters": []}
        self._list_cache = list(seen_paths.values())
        return self._list_cache

    def load(self, name: str, _loading_stack: set | None = None, **params) -> str:
        """Load a skill by frontmatter name or directory name. Later directories take priority.

        Optional keyword arguments are substituted into {param_name} placeholders
        in the skill body. Parameters defined in frontmatter with defaults are
        used when not provided by the caller.

        Auto-loads dependency summaries declared in frontmatter 'requires' field.
        Appends 'suggests' hints for related skills.
        """
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            raise FileNotFoundError(f"Skill '{name}' not found in: {self._dirs}")
        meta, body = self._parse_file(skill_file)
        skill_name = meta.get("name", name)

        # Auto-load dependency summaries with circular dependency detection
        if _loading_stack is None:
            _loading_stack = set()
        _loading_stack.add(name)

        requires = meta.get("requires", [])
        if requires and isinstance(requires, list):
            dep_hints = []
            for dep_name in requires:
                if dep_name in _loading_stack:
                    continue
                # Prefer summary over full content for dependencies
                summary = self.load_summary(dep_name)
                if summary:
                    dep_hints.append(f"<dependency name=\"{dep_name}\" type=\"summary\">\n{summary}\n</dependency>")
                else:
                    try:
                        dep_content = self.load(dep_name, _loading_stack=_loading_stack, **params)
                        dep_hints.append(dep_content)
                    except FileNotFoundError:
                        pass
            if dep_hints:
                body = "\n\n".join(dep_hints) + "\n\n" + body

        # Append suggests hints (lightweight — just names and descriptions)
        suggests = meta.get("suggests", [])
        if suggests and isinstance(suggests, list):
            available = {s["name"] for s in self.list_skills()}
            valid_suggests = [s for s in suggests if s in available]
            if valid_suggests:
                hints = ", ".join(f"`{s}`" for s in valid_suggests)
                body += f"\n\n---\nRelated skills (load if needed): {hints}"

        _loading_stack.discard(name)

        param_defs = meta.get("parameters", [])
        if isinstance(param_defs, list):
            for pdef in param_defs:
                if isinstance(pdef, dict):
                    pname = pdef.get("name", "")
                    if pname and pname not in params and "default" in pdef:
                        params[pname] = pdef["default"]

        for k, v in params.items():
            body = body.replace(f"{{{k}}}", str(v))

        return f"<skill name=\"{skill_name}\">\n{body}\n</skill>"

    def get_effects(self, name: str) -> Dict:
        """Get the 'effects' declaration from skill frontmatter.

        Returns a dict that may contain:
          - mode: str — mode flag to set on the agent (e.g. "porting")
          - companion_skills: list[str] — skills to auto-load alongside
        Returns empty dict if no effects declared.
        """
        meta = self.get_meta(name)
        return meta.get("effects", {}) or {}

    def get_meta(self, name: str) -> Dict:
        """Get skill frontmatter metadata without loading full content. Cached."""
        if name in self._meta_cache:
            return self._meta_cache[name]
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            return {}
        try:
            meta, _ = self._parse_file(skill_file)
            self._meta_cache[name] = meta
            return meta
        except Exception:
            return {}

    def get_dependency_closure(self, names: list, _seen: set | None = None) -> list:
        """Return a topologically sorted list of skill names to load, including
        all transitive `requires` and `suggests` of the given skills.

        `requires` are loaded first (strong deps), then the skill itself,
        then `suggests` (weak deps). Circular references are skipped.
        """
        if _seen is None:
            _seen = set()
        result = []
        for name in names:
            if name in _seen:
                continue
            _seen.add(name)
            meta = self.get_meta(name)
            if not meta:
                continue
            # Depth-first: load requires first
            reqs = meta.get("requires", []) or []
            for dep in reqs:
                if dep not in _seen:
                    sub_result = self.get_dependency_closure([dep], _seen)
                    result.extend(sub_result)
            # Then the skill itself (if not already added by depends chain)
            # We add the skill name separately so the caller loads it
            if name not in [r for r in result]:
                result.append(name)
            # Then suggests (weak deps, loaded after)
            suggs = meta.get("suggests", []) or []
            for sug in suggs:
                if sug not in _seen:
                    sub_result = self.get_dependency_closure([sug], _seen)
                    result.extend(sub_result)
        return result

    def load_summary(self, name: str) -> str | None:
        """Load SUMMARY.md for a skill if it exists. Returns None if no summary available."""
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            return None
        skill_dir = os.path.dirname(skill_file)
        summary_file = os.path.join(skill_dir, "SUMMARY.md")
        if not os.path.isfile(summary_file):
            return None
        with open(summary_file, "r", encoding="utf-8") as f:
            return f.read()

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
