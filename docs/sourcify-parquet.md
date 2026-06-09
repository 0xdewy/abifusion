# Sourcify Parquet Exports

Sourcify publishes public v2 database exports as parquet files. This repo uses
those exports to build larger ABI reconstruction datasets with verified ABIs and
runtime bytecode.

Official export shape:

```text
https://export.sourcify.dev/v2/<table>/<file>.parquet
```

Local repo layout:

```text
data/sourcify/
  code/
    code_0_100000.parquet
    ...
  contracts/
    contracts_0_1000000.parquet
    ...
  contract_deployments/
    contract_deployments_0_1000000.parquet
    ...
  compiled_contracts/
    compiled_contracts_0_10000.parquet
    ...
```

## Required Tables

The dataset builder needs four Sourcify tables:

| Table | Used For |
| --- | --- |
| `contract_deployments` | Address, chain ID, and contract ID |
| `contracts` | Runtime code hash for each contract ID |
| `code` | Runtime bytecode keyed by code hash |
| `compiled_contracts` | Compilation artifacts, including ABIs |

Other exported tables are useful for source-code research, but not required for
`scripts/build_final_dataset.py`.

## Recommended Workflow

First, verify that the export listing is reachable:

```bash
python scripts/download_sourcify_parquet.py \
  --max-files-per-table 1 \
  --list-only
```

Expected output includes one parquet key for each default table:

```text
v2/contracts/contracts_0_1000000.parquet
v2/contract_deployments/contract_deployments_0_1000000.parquet
v2/compiled_contracts/compiled_contracts_0_10000.parquet
v2/code/code_0_100000.parquet
```

Download the required tables:

```bash
python scripts/download_sourcify_parquet.py
```

Build a project-ready dataset:

```bash
python scripts/build_final_dataset.py \
  --sourcify-dir data/sourcify \
  --max-contracts 100000 \
  --output data/sourcify_100k_final.parquet
```

Run a smaller smoke-test build:

```bash
python scripts/download_sourcify_parquet.py --max-files-per-table 1
python scripts/build_final_dataset.py --test-mode
```

## Downloader Commands

List files without downloading:

```bash
python scripts/download_sourcify_parquet.py --list-only
```

Download only selected tables:

```bash
python scripts/download_sourcify_parquet.py \
  --tables contracts,contract_deployments,compiled_contracts,code
```

Download every exported Sourcify table:

```bash
python scripts/download_sourcify_parquet.py --tables all
```

Download into a custom directory:

```bash
python scripts/download_sourcify_parquet.py \
  --output-dir /tmp/sourcify-parquet
```

Resume behavior is default: existing files are skipped. To re-download:

```bash
python scripts/download_sourcify_parquet.py --overwrite
```

## Dataset Builder

The builder accepts the v2 folder layout:

```bash
python scripts/build_final_dataset.py --sourcify-dir data/sourcify
```

It also accepts the older single-file layout for compatibility:

```text
data/sourcify/code.parquet
data/sourcify/contracts.parquet
data/sourcify/contract_deployments.parquet
data/sourcify/compiled_contracts.parquet
```

Output columns:

| Column | Description |
| --- | --- |
| `address` | Deployed contract address |
| `bytecode` | Runtime bytecode as `0x` hex |
| `bytecode_length` | Runtime bytecode hex length without `0x` |
| `abi` | Verified ABI from compilation artifacts |
| `has_abi` | Whether an ABI was extracted |
| `abi_length` | Number of ABI entries |
| `chain_id` | Deployment chain ID |
| `runtime_code_hash` | Runtime bytecode hash |
| `contract_id` | Sourcify contract ID |

## Direct Bucket Sync

If you prefer AWS CLI-style syncing, use Sourcify's public GCS-backed bucket:

```bash
aws s3 sync \
  s3://sourcify-production-parquet-export/v2/ \
  data/sourcify \
  --endpoint-url https://storage.googleapis.com \
  --no-sign-request
```

For this repo, syncing only the four required table prefixes is usually enough.

## Troubleshooting

**Name resolution or connection error**

The downloader needs network access to `export.sourcify.dev`. In restricted
environments, run it where outbound HTTPS is allowed.

**Missing table error from `build_final_dataset.py`**

Download the missing table:

```bash
python scripts/download_sourcify_parquet.py --tables <table_name>
```

**Out-of-memory while building**

Start with fewer parquet files:

```bash
python scripts/download_sourcify_parquet.py --max-files-per-table 1
python scripts/build_final_dataset.py --test-mode
```

Then increase the number of downloaded files or run the build on a larger
machine.
