#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2026 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2026	Abdelkrime Aries <kariminfo0@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, TypedDict

@dataclass
class Work:
    id: str
    title: Optional[str]
    year: Optional[str]
    abstract: str
    link: Optional[str]
    authors: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    language: Optional[str] = None
    venue: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuthorCandidate:
    id: str
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    interests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

