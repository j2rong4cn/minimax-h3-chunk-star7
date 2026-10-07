param([string]$CudaRoot = "", [string]$VisualStudioRoot = "")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $CudaRoot) {
    $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
}
if (-not $VisualStudioRoot) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $VisualStudioRoot = & $vswhere -latest -version "[16.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
$vcvars = Join-Path $VisualStudioRoot "VC\Auxiliary\Build\vcvars64.bat"
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$output = Join-Path $projectRoot "bin\win_amd64\star7_h3_preprocess_v1.dll"
$source = Join-Path $PSScriptRoot "h3_preprocess.cu"
$command = 'call "{0}" && "{1}" -shared -O3 --use_fast_math -std=c++17 -arch=sm_75 --cudart static -Xcompiler=/MT -Xcompiler=/O2 -o "{2}" "{3}"' -f $vcvars, $nvcc, $output, $source
& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "H3 preprocessing CUDA build failed" }
Remove-Item -LiteralPath ([IO.Path]::ChangeExtension($output, '.lib')),([IO.Path]::ChangeExtension($output, '.exp')) -ErrorAction SilentlyContinue
@{ abi_version = 1; file = 'win_amd64/star7_h3_preprocess_v1.dll'; sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant() } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'bin\h3_preprocess_manifest.json') -Encoding utf8
