# Hard Cases Analysis: Selectors with No 4byte AND No evmole

Total hard cases: **244**
Unique selectors: **86**

## Function family breakdown:
  - **Uniswap V3 callbacks**: 126
  - **other**: 65
  - **NFT setters**: 46
  - **SVG generators**: 4
  - **execute helpers**: 3

## Selector frequency:
  - Repeated selectors: **29** (187 occurrences)
  - Unique-only selectors: **57**

## Top repeated selectors:
  - `468ead2c` ×18: beforeSwap(address,tuple,tuple,bytes)
  - `bc29bafc` ×12: afterAddLiquidity(address,tuple,tuple,int256,int256,bytes)
  - `1f29cf9d` ×12: afterDonate(address,tuple,uint256,uint256,bytes)
  - `b6d4944a` ×12: afterInitialize(address,tuple,uint160,int24)
  - `606e8192` ×12: afterRemoveLiquidity(address,tuple,tuple,int256,int256,bytes)
  - `c13d1c69` ×12: afterSwap(address,tuple,tuple,int256,bytes)
  - `b772b8cc` ×12: beforeAddLiquidity(address,tuple,tuple,bytes)
  - `d950bd74` ×12: beforeDonate(address,tuple,uint256,uint256,bytes)
  - `ebe1cdaf` ×12: beforeInitialize(address,tuple,uint160)
  - `5cb32d10` ×12: beforeRemoveLiquidity(address,tuple,tuple,bytes)
  - `a660668a` ×4: generateSvg(tuple)
  - `71edd89d` ×4: setAccessories(tuple[])
  - `d8234121` ×4: setBody(tuple[])
  - `d33aed88` ×4: setEyes(tuple[])
  - `2201fddb` ×4: setGround(tuple[])
  - `49b69b3f` ×4: setHair(tuple[])
  - `5d5ce9c9` ×4: setHorn(tuple[])
  - `0bf1d8f2` ×4: setLegsBack(tuple[])
  - `fed15348` ×4: setLegsFront(tuple[])
  - `8728fc6e` ×4: setTail(tuple[])

## Sample unique selectors (first 15):
  - `17ca339c`: execute(tuple[],tuple[],address,address)
  - `c7e2c207`: executeAndSendBalance(tuple[],tuple,address)
  - `d5fffb7f`: executeFinalCall(tuple,tuple,address)
  - `262b74c9`: swapExactInputSingle(tuple)
  - `e9e3ea46`: addLiquidity(tuple,int24,int24,uint256)
  - `9128d1b1`: claimFees(tuple,int24,int24)
  - `9148af05`: positionLiquidity(tuple,int24,int24)
  - `6f5820fa`: removeLiquidity(tuple,int24,int24,uint128)
  - `b503dfa3`: ccipReceive(tuple)
  - `6971af8f`: updateState(tuple)
  - `479476d8`: updateStateAndDeposit(address,address,tuple)
  - `8d4c30c0`: updateStateAndDepositAndMintOsToken(address,uint256,address,tuple)
  - `1aff7a92`: batchUpdatePhases(uint256[],tuple[])
  - `f63c8122`: deployPhases(tuple[])
  - `95498386`: updatePhase(uint256,tuple)

## Selector frequency distribution:
  - ×18: 1 selectors
  - ×12: 9 selectors
  - ×4: 11 selectors
  - ×3: 1 selectors
  - ×2: 7 selectors
  - ×1: 57 selectors
