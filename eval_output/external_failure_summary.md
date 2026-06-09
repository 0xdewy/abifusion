# ABI Reconstructor Failure Summary

**Date:** 2026-06-09

## Overview

| Metric | Value |
|--------|-------|
| Total functions | 9249 |
| Correct | 9101 |
| Failed | 148 |
| Accuracy | 98.4% |
| Split seed | 42 |
| Held-out fraction | 0.2 |

## Failure Categories

| Category | Count | % |
|----------|-------|-------|
| selector_missing_from_bytecode | 136 | 1.5% |
| known_selector_table_miss | 12 | 0.1% |

## Addressability

| Class | Count | % |
|----------|-------|-------|
| not_addressable_from_bytecode | 136 | 1.5% |
| deterministically_addressable | 12 | 0.1% |

## Source Tier Breakdown

| Source | Count | % |
|----------|-------|-------|
| unknown | 148 | 1.6% |

## Confidence Breakdown

| Confidence | Count | % |
|----------|-------|-------|
| unknown | 148 | 1.6% |
