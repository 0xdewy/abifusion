# Gap Analysis — ABI Reconstructor

**20 contracts · 480 ground-truth functions**
**462 correct (96.2%) · 18 failures (3.8%)**

## Failure Breakdown

| Category | Count | % of Gap | Fixable? |
|---|---|---|---|
| A. Selector not found | 17 | 94.4% | Maybe (parser improvement) |
| B. 4byte DB has no entry | 0 | 0.0% | Yes (ML or DB expansion) |
| C. 4byte DB collision (multiple signatures) | 0 | 0.0% | Yes (signature disambiguation) |
| D. Proxy contract (structural) | 1 | 5.6% | No (structural) |
| E. Type mismatch despite DB hit | 0 | 0.0% | Bug — investigate |

## Fixability Summary

- **Fixable** (DB miss or collision): 0 (0.0% of gap) — these can be resolved by ML type prediction or signature disambiguation
- **Maybe fixable** (selector extraction): 17 (94.4% of gap) — these selectors exist in bytecode but our parser misses them
- **Structural** (proxy contracts): 1 (5.6% of gap) — proxy bytecode selectors ≠ logical ABI, not fixable without implementation resolution
- **Bugs**: 0 (0.0% of gap) — type mismatch despite single DB entry, needs investigation

## Projected improvement

- After selector extraction improvement: → 0.998

## A. Selector not found (17 examples)

- `SwapRouter.exactInput(tuple)`
  - selector `80fb3ad6`

- `SwapRouter.exactInputSingle(tuple)`
  - selector `5d76b977`

- `SwapRouter.exactOutput(tuple)`
  - selector `d42bbb58`

- `SwapRouter.exactOutputSingle(tuple)`
  - selector `5bd7800f`

- `AggregationRouterV5.cancelOrder(tuple)`
  - selector `3089c27d`

- `AggregationRouterV5.checkPredicate(tuple)`
  - selector `513000d6`

- `AggregationRouterV5.fillOrder(tuple,bytes,bytes,uint256,uint256,uint256)`
  - selector `a06869de`

- `AggregationRouterV5.fillOrderRFQ(tuple,bytes,uint256)`
  - selector `fa1914c1`

  ... and 9 more like this


## D. Proxy contract (structural) (1 examples)

- `UniswapV2Router02.addLiquidity(address,address,uint256,uint256,uint256,uint256,address,uint256)`
  - selector `e8e33700`
  - reconstructed as `(address,uint256)`
  - proxy reason: delegatecall in bytecode
