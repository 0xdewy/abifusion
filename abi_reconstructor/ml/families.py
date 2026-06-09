"""Function family taxonomy for zero-shot name prediction.

Instead of predicting exact function names (1,656 classes), we predict
function families (~20-50 classes). This dramatically reduces the classification
difficulty and enables generalization to unseen function names.

Families are semantic groupings based on:
- Token standards (ERC20, ERC721/1155, ERC20Votes)
- Protocol types (AMM/DEX, Lending, Bridge)
- Access control patterns (Ownable, AccessControl, Pausable)
- Common operations (token transfers, liquidity management, governance)
"""

from __future__ import annotations

from typing import Dict, List

FAMILY_NAMES: List[str] = [
    "ERC20_BASIC",      # transfer, approve, balanceOf, allowance
    "ERC20_DETAILED",   # decimals, name, symbol, totalSupply
    "ERC20_RESTRICTED", # transferOwnership, renounceOwnership, pause, unpause
    "ERC20_BURN",       # burn, burnFrom
    "ERC20_APPROVE",    # increaseAllowance, decreaseAllowance, permit, DOMAIN_SEPARATOR, nonces
    "ERC721_BASIC",     # ownerOf, getApproved, isApprovedForAll, setApprovalForAll
    "ERC721_METADATA",  # tokenURI, metadata, updateMetadata, imageUrl, updateImage
    "ERC721_TRANSFER",  # safeTransferFrom, onERC721Received
    "ERC1155",          # royaltyInfo
    "ERC20VOTES",       # delegate, delegateBySig, delegates, checkpoints, numCheckpoints, getPastVotes, getVotes, CLOCK_MODE
    "AMM_LIQUIDITY",    # addLiquidity, removeLiquidity, remove_liquidity, getReserves, skim, sync
    "AMM_SWAP",         # swap, beforeSwap, afterSwap, manualSwap
    "AMM_CORE",         # mint, burn, factory, uniswapV2Pair, bondingCurveFactory
    "AMM_CALLBACK",     # beforeAddLiquidity, afterAddLiquidity, beforeRemoveLiquidity, afterRemoveLiquidity, beforeDonate, afterDonate, beforeInitialize, afterInitialize
    "ACCESS_OWNABLE",   # owner, pendingOwner, acceptOwnership, requestOwnershipHandover, completeOwnershipHandover, cancelOwnershipHandover, ownershipHandoverExpiresAt
    "ACCESS_ADMIN",     # admin, originalAdmin, updateAdmin, setAdmin, transferOwnership, renounceOwnership
    "ACCESS_CONTROL",   # hasRole, grantRole, revokeRole, renounceRole, DEFAULT_ADMIN_ROLE
    "PAUSABLE",         # pause, unpause, paused
    "SETTINGS",         # setBaseURI, setTreasury, setEthRecipient, setMode, setTaxSwapMult, setHookPermissions, setAuthority, setDelay, updateDelay, delay
    "TOKEN_SETTINGS",   # transferTaxDisabled, disableTransferTax, openTrading, enableTrading, removeLimits, reduceFee
    "WALLET",           # deposit, withdraw, withdrawal, withdrawEth, rescueERC20, rescueTokens, rescue
    "CROSSCHAIN",      # ccipReceive, ccipSend, transferCrossChain
    "GOVERNANCE",       # proposal, vote, castVote, execute, queue, propose, TIMELOCK
    "TOKEN_OPERATIONS", # mint, transfer, _metaUrl, metaUrl, tokenName, tokenSymbol, allData, mirror
    "CLOCK",            # clock, CLOCK_MODE, allData, context, TOTAL_SUPPLY, PERMIT2, POSM
    "VOTES",            # checkpoints, delegates, delegateBySig, getPastTotalSupply, getVotes, originalAdmin
    "BOT", # isBot, addBots, delBots
    "TRADING",          # frontrun, openTrading, enableTrading, bondingCurveFactory, poolManager
    "UPGRADE",          # upgradeToAndCall, upgradeAndCall, implementation, version, init, initialize, UPGRADE_INTERFACE_VERSION
    "PARAMETERS",       # _maxTaxSwap, _maxTxAmount, _maxWalletSize, _taxSwapThreshold, _metaUrl, _mode, TAX_BPS, BPS_DENOM, SUPPLY, MIN_DELAY, MAX_DELAY, UNIT, TICK_SPACING, ALLOWED_TOKEN, REGISTRY, DELAY, MIN_DELAY
    "GENERAL",          # get, set, update, create, execute, verify, start, stop, pause, unpause, mint, transfer, rescue, isVerified, initialized, hook, treasury, ethRecipient, authority, contractURI, teamToken, VERSION, VERSION
    "CALLBACK",         # uniswapV2Pair, bondingCurveFactory, poolManager, trenchManager
    "UNKNOWN",          # fallback for unmapped names
]

FAMILY_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FAMILY_NAMES)}


def get_family(name: str) -> int:
    name_lower = name.lower()
    if name_lower in (
        "transfer", "transferfrom", "approve", "balanceof", "allowance",
        "transferfrom", "safe_transferfrom",
    ):
        return FAMILY_INDEX["ERC20_BASIC"]
    if name_lower in (
        "decimals", "name", "symbol", "totalsupply", "symbol",
        "max_supply", "total_supply", "totalshares",
    ):
        return FAMILY_INDEX["ERC20_DETAILED"]
    if name_lower in (
        "transferownership", "renounceownership", "pause", "unpause", "paused",
    ):
        return FAMILY_INDEX["ERC20_RESTRICTED"]
    if name_lower in ("burn", "burnfrom", "burnfrom"):
        return FAMILY_INDEX["ERC20_BURN"]
    if name_lower in (
        "increaseallowance", "decreaseallowance", "permit", "domain_separator",
        "nonces", "delegation", "permit", "eip712domain",
    ):
        return FAMILY_INDEX["ERC20_APPROVE"]
    if name_lower in (
        "ownerof", "getapproved", "isapprovedforall", "setapprovalforall",
    ):
        return FAMILY_INDEX["ERC721_BASIC"]
    if name_lower in (
        "tokenuri", "metadata", "updatemetadata", "imageurl", "updateimage",
        "baseuri", "setbaseuri",
    ):
        return FAMILY_INDEX["ERC721_METADATA"]
    if name_lower in ("safetransferfrom", "onerc721received"):
        return FAMILY_INDEX["ERC721_TRANSFER"]
    if name_lower in ("royaltyinfo",):
        return FAMILY_INDEX["ERC1155"]
    if name_lower in (
        "delegate", "delegatebysig", "delegates", "checkpoints", "numcheckpoints",
        "getpastvotes", "getvotes", "clock_mode", "allstorage", "getpasttotalsupply",
        "totalminted",
    ):
        return FAMILY_INDEX["ERC20VOTES"]
    if name_lower in (
        "addliquidity", "removeliquidity", "remove_liquidity", "getreserves",
        "skim", "sync",
    ):
        return FAMILY_INDEX["AMM_LIQUIDITY"]
    if name_lower in ("swap", "beforeswap", "afterswap", "manualswap"):
        return FAMILY_INDEX["AMM_SWAP"]
    if name_lower in ("mint", "burn", "factory", "uniswapv2pair", "bondingcurvefactory", "uniswapv2"):
        return FAMILY_INDEX["AMM_CORE"]
    if name_lower in (
        "beforeaddliquidity", "afteraddliquidity",
        "beforeremoveliquidity", "afterremoveliquidity",
        "beforedonate", "afterdonate",
        "beforeinitialize", "afterinitialize",
    ):
        return FAMILY_INDEX["AMM_CALLBACK"]
    if name_lower in (
        "owner", "pendingowner", "acceptownership", "requestownershiphandover",
        "completeownershiphandover", "cancelownershiphandover",
        "ownershiphandoverexpiresat",
    ):
        return FAMILY_INDEX["ACCESS_OWNABLE"]
    if name_lower in (
        "admin", "originaladmin", "updateadmin", "setadmin",
    ):
        return FAMILY_INDEX["ACCESS_ADMIN"]
    if name_lower in (
        "hasrole", "grantrole", "revokerole", "renouncesrole",
        "default_admin_role", "timelock",
    ):
        return FAMILY_INDEX["ACCESS_CONTROL"]
    if name_lower in ("pause", "unpause", "paused"):
        return FAMILY_INDEX["PAUSABLE"]
    if name_lower in (
        "setbaseuri", "settreasury", "setethrecipient", "setmode",
        "settaxswapmult", "sethookpermissions", "setauthority", "setdelay",
        "updatedelay", "delay", "delayupdatetime", "implementationupdatetime",
        "newdelay", "newimplementation", "submitdelay", "submitimplementation",
    ):
        return FAMILY_INDEX["SETTINGS"]
    if name_lower in (
        "transfertaxdisabled", "disabletransfertax", "opentrading",
        "enabletrading", "removelimits", "reducefee",
    ):
        return FAMILY_INDEX["TOKEN_SETTINGS"]
    if name_lower in (
        "deposit", "withdraw", "withdrawal", "withdraweth",
        "rescueerc20", "rescuetokens", "rescue", "depositrequests",
    ):
        return FAMILY_INDEX["WALLET"]
    if name_lower in ("ccipreceive", "ccipsend", "transfercrosschain"):
        return FAMILY_INDEX["CROSSCHAIN"]
    if name_lower in (
        "proposal", "vote", "castvote", "execute", "queue", "propose",
    ):
        return FAMILY_INDEX["GOVERNANCE"]
    if name_lower in (
        "mint", "transfer", "_metaurl", "metaurl", "tokenname",
        "tokensymbol", "alldata", "mirror",
    ):
        return FAMILY_INDEX["TOKEN_OPERATIONS"]
    if name_lower in (
        "clock", "clock_mode", "allstorage", "totalsupply", "permit2", "posm",
    ):
        return FAMILY_INDEX["CLOCK"]
    if name_lower in (
        "checkpoints", "delegates", "delegatedbysig", "getpasttotalsupply",
        "getvotes", "originaladmin",
    ):
        return FAMILY_INDEX["VOTES"]
    if name_lower in ("isbot", "addbots", "delbots"):
        return FAMILY_INDEX["BOT"]
    if name_lower in (
        "frontrun", "opentrading", "enabletrading", "bondingcurvefactory",
        "poolmanager",
    ):
        return FAMILY_INDEX["TRADING"]
    if name_lower in (
        "upgradetoandcall", "upgradeandcall", "implementation",
        "version", "init", "initialize", "upgrade_interface_version",
    ):
        return FAMILY_INDEX["UPGRADE"]
    if name_lower in (
        "_maxtaxswap", "_maxtxamount", "_maxwalletsize", "_taxswapthreshold",
        "_metaurl", "_mode", "tax_bps", "bps_denom", "supply", "min_delay",
        "max_delay", "unit", "tick_spacing", "allowed_token", "registry",
        "delay", "mode_normal", "mode_transfer_controlled", "mode_transfer_restricted",
        "trenchmanager", "poolmanager", "start", "globalticklower", "globaltickupper",
        "hookpositiontokenid",
    ):
        return FAMILY_INDEX["PARAMETERS"]
    if name_lower in (
        "get", "set", "update", "create", "execute", "verify", "start",
        "stop", "isverified", "initialized", "hook", "treasury",
        "ethrecipient", "authority", "contracturi", "teamtoken", "version",
        "context",
    ):
        return FAMILY_INDEX["GENERAL"]
    if name_lower in (
        "gethookpermissions", "sethookpermissions", "hookpermissions",
    ):
        return FAMILY_INDEX["SETTINGS"]
    if name_lower in (
        "executerequestbykeeper", "ingressvitals", "nextrequestid",
        "poolkey", "unlockcallback", "seeded", "isstarted", "deployer",
        "proxiableuuid", "metadatauri", "platformethwallet", "bps",
        "mpegperupepg", "mpeg_swap_unit", "initial_supply", "tokenbyindex",
        "tokenofownerbyindex", "uri", "totalshares", "depositrequests",
        "globalticklower", "globaltickupper", "hookpositiontokenid",
        "start", "stop", "isverified", "initialized", "hook", "treasury",
        "ethrecipient", "authority", "contracturi", "teamtoken", "version",
        "context", "allstorage",
    ):
        return FAMILY_INDEX["GENERAL"]
    if name_lower in (
        "uniswapv2pair", "bondingcurvefactory", "poolmanager", "trenchmanager",
    ):
        return FAMILY_INDEX["CALLBACK"]
    if name_lower in ("supportsinterface",):
        return FAMILY_INDEX["ERC20_DETAILED"]
    if name_lower in ("uniswapv2",):
        return FAMILY_INDEX["AMM_CORE"]
    return FAMILY_INDEX["UNKNOWN"]


def get_family_name(family_idx: int) -> str:
    return FAMILY_NAMES[family_idx] if family_idx < len(FAMILY_NAMES) else "UNKNOWN"
