param(
    [Parameter(Mandatory=$true)][string]$CutlassInclude,
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
    [string]$VisualStudioRoot = ""
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath (Join-Path $CutlassInclude 'cutlass\cutlass.h'))) {
    throw 'Pass the include directory of NVIDIA CUTLASS 4.2.0 to -CutlassInclude'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $VisualStudioRoot) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $VisualStudioRoot = & $vswhere -latest -version '[16.0,18.0)' -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
$vcvars = Join-Path $VisualStudioRoot 'VC\Auxiliary\Build\vcvars64.bat'
$nvcc = Join-Path $CudaRoot 'bin\nvcc.exe'
$native = Join-Path $PSScriptRoot 'w4a8_native'
$overlay = Join-Path $native 'cutlass_overlay\include'
$cccl = Join-Path $CudaRoot 'include\cccl'
$source = Join-Path $native 'abi.cu'
$output = Join-Path $projectRoot 'bin\win_amd64\star7_w4a8_sm75_v1.dll'
$command = 'call "{0}" && "{1}" -shared -O3 -std=c++20 -arch=sm_75 --cudart static -Xcompiler=/MT -Xcompiler=/O2 -Xcompiler=/EHsc -Xcompiler=/bigobj -Xcompiler=/Zc:preprocessor -Xcompiler=/permissive -I"{2}" -I"{3}" -I"{4}" -I"{5}" -o "{6}" "{7}"' -f $vcvars,$nvcc,$native,$overlay,$CutlassInclude,$cccl,$output,$source
& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw 'W4A8 CUDA build failed' }
Remove-Item -LiteralPath ([IO.Path]::ChangeExtension($output,'.lib')),([IO.Path]::ChangeExtension($output,'.exp')) -ErrorAction SilentlyContinue
@{ abi_version = 1; windows_x64 = @{ file = 'win_amd64/star7_w4a8_sm75_v1.dll'; sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant(); size = (Get-Item -LiteralPath $output).Length } } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'bin\w4a8_manifest.json') -Encoding utf8
