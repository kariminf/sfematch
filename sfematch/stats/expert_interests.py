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

import re
import sys
import statistics
from collections import defaultdict

EXPERT_ID_RE = re.compile(r'^XPRT\d+$', re.IGNORECASE)


def parse_interests_file(path):
    """
    Returns: dict[type_name] -> list[int]  (one count of interests per expert
    that had that type line present, in the order encountered)
    """
    type_counts = defaultdict(list)
    current_expert = None
    # Track which types we've already seen for the current expert, in case
    # a type appears more than once for the same expert (unlikely, but safe).
    seen_types_for_current = set()

    with open(path, encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n').rstrip('\r')
            stripped = line.strip()

            if not stripped:
                continue

            if EXPERT_ID_RE.match(stripped):
                current_expert = stripped
                seen_types_for_current = set()
                continue

            if current_expert is None:
                # Line encountered before any expert ID; skip it.
                continue

            if ':' not in stripped:
                # Not a "type: interests" line; skip.
                continue

            type_name, _, value = stripped.partition(':')
            type_name = type_name.strip()
            value = value.strip()

            if not type_name:
                continue

            # Normalize the type name for consistent grouping (case-insensitive),
            # but keep the first-seen casing for display.
            key = type_name.lower()

            if value == '':
                count = 0
            else:
                interests = [i.strip() for i in value.split(';')]
                interests = [i for i in interests if i]  # drop empty entries
                count = len(interests)

            type_counts[key].append(count)
            seen_types_for_current.add(key)

    return type_counts


def compute_interests_stats(type_counts):
    """
    Returns: dict[type_name] -> dict of stats
    """
    results = {}
    for type_name, counts in type_counts.items():
        if not counts:
            continue
        results[type_name] = {
            'min_number_of_interests': min(counts),
            'number_of_experts_without_interests': sum(1 for c in counts if c == 0),
            'max_number_of_interests': max(counts),
            'avg_number_of_interests': round(statistics.mean(counts), 2),
            'median_number_of_interests': statistics.median(counts),
            'total_experts_with_this_type': len(counts),
        }
    return results


def print_interests_table(stats):
    if not stats:
        print("No data found.")
        return

    headers = [
        'type',
        'min_number_of_interests',
        'number_of_experts_without_interests',
        'max_number_of_interests',
        'avg_number_of_interests',
        'median_number_of_interests',
    ]

    rows = []
    for type_name in sorted(stats.keys()):
        s = stats[type_name]
        rows.append([
            type_name,
            s['min_number_of_interests'],
            s['number_of_experts_without_interests'],
            s['max_number_of_interests'],
            s['avg_number_of_interests'],
            s['median_number_of_interests'],
        ])

    col_widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(len(headers))]

    def fmt_row(row):
        return '  '.join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))

    print(fmt_row(headers))
    print('  '.join('-' * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def write_interests_csv(stats, out_path):
    import csv
    headers = [
        'type',
        'min_number_of_interests',
        'number_of_experts_without_interests',
        'max_number_of_interests',
        'avg_number_of_interests',
        'median_number_of_interests',
    ]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for type_name in sorted(stats.keys()):
            s = stats[type_name]
            writer.writerow([
                type_name,
                s['min_number_of_interests'],
                s['number_of_experts_without_interests'],
                s['max_number_of_interests'],
                s['avg_number_of_interests'],
                s['median_number_of_interests'],
            ])
    print(f"\nCSV written to: {out_path}")


