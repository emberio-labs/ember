"""Скиллы агента по стандарту Agent Skills (agentskills.io).

Публичные точки входа: ``Skill`` (модель данных), ``parse_skill`` /
``scan_skills`` (лояльное чтение) и ``SkillStore`` (каталог скиллов плюс
инструменты ``read_skill`` / ``save_skill`` для ``Agent``).
"""

from ember.skills.base import Skill
from ember.skills.loader import parse_skill, scan_skills
from ember.skills.tools import SkillStore

__all__ = ["Skill", "SkillStore", "parse_skill", "scan_skills"]
