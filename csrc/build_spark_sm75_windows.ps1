param(
    [string]$CudaRoot = "",
    [string]$VisualStudioRoot = "",
    [string]$OutputName = "star7_spark_sm75_v3.dll",
    [int]$MaxRegisters = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($OutputName -notmatch '^star7_spark_sm75_[a-zA-Z0-9_]+\.dll$') {
    throw "OutputName must be a plain Spark SM75 DLL filename"
}
$source = Join-Path $PSScriptRoot "spark_sm75.cu"
$outputDir = Join-Path $projectRoot "bin\win_amd64"
$output = Join-Path $outputDir $OutputName

if (-not $CudaRoot) {
    $nvccCommand = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($nvccCommand) {
        $CudaRoot = Split-Path -Parent (Split-Path -Parent $nvccCommand.Source)
    } else {
        $candidates = Get-ChildItem `
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" `
            -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        if (-not $candidates) { throw "CUDA Toolkit was not found" }
        $CudaRoot = $candidates[0].FullName
    }
}

if (-not $VisualStudioRoot) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) { throw "vswhere.exe was not found" }
    # CUDA 13.0 accepts VS 2019/2022 but rejects the newer VS 2026 host
    # compiler.  Prefer the newest installation from the supported range.
    $VisualStudioRoot = & $vswhere -latest -version "[16.0,18.0)" -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $VisualStudioRoot) { throw "Visual Studio C++ Build Tools were not found" }
}

$vcvars = Join-Path $VisualStudioRoot "VC\Auxiliary\Build\vcvars64.bat"
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $vcvars)) { throw "vcvars64.bat was not found: $vcvars" }
if (-not (Test-Path -LiteralPath $nvcc)) { throw "nvcc.exe was not found: $nvcc" }
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$registerFlag = if ($MaxRegisters -gt 0) { "--maxrregcount=$MaxRegisters" } else { "" }
$command = 'call "{0}" && "{1}" -shared -O3 --use_fast_math -std=c++17 -arch=sm_75 --cudart static -Xcompiler=/MT -Xcompiler=/O2 -Xptxas=-v {2} -o "{3}" "{4}"' -f `
    $vcvars, $nvcc, $registerFlag, $output, $source
& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "NVCC failed with exit code $LASTEXITCODE" }
$manifestPath = Join-Path $projectRoot 'bin\spark_sm75_manifest.json'
$manifest = if (Test-Path -LiteralPath $manifestPath) {
    Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
} else { @{ abi = 1; architecture = 'sm75'; platforms = @{} } }
$manifest['cuda'] = (Split-Path -Leaf $CudaRoot).TrimStart('v')
$manifest['platforms']['windows_x64'] = @{
    file = "win_amd64/$OutputName"
    sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
}
[System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8),
    [System.Text.UTF8Encoding]::new($false))
Write-Host "Built $output"
