<#
.SYNOPSIS
gen_dialect_maps: predict dialect classification and generate dialect maps

.DESCRIPTION
Predict dialect classification based on pre-defined rules and annotations, and generate isogloss maps and dialect partition maps

.PARAMETER Help
Show help

.PARAMETER Debug
Enable debug log

.PARAMETER RuleFile
Rule file in JSON format (default: rules.json)

.PARAMETER AnnotationFile
Annotation file in CSV format (default: annotations.csv)

.PARAMETER ComplianceFile
Precomputed compliance file in CSV format, if not specified, compute from specified datasets

.PARAMETER Datasets
Datasets to compute rule compliances and classify (default: CCR)

.PARAMETER BackgroundFile
Background raster image file (default: HYP_50M_SR_W\HYP_50M_SR_W.tif)

.PARAMETER GeographyFile
Geography file in GeoJSON/Shapefile format (default: ne_50m_land\ne_50m_land.shp)

.PARAMETER Extent
Extent of output maps, or coverage of dialect points

.PARAMETER Size
Output image size

.PARAMETER OutputDir
Output folder (default: current working folder)
#>

param(
    [switch]$Help,
    [switch]$Debug,
    [string]$RuleFile = 'rules.json',
    [string]$AnnotationFile = 'annotations.csv',
    [string]$ComplianceFile,
    [string]$Datasets = @('CCR'),
    [string]$BackgroundFile,
    [string]$GeographyFile,
    [string]$Extent,
    [string]$Size,
    [string]$OutputDir = (Get-Location)
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path
    exit 0
}

$LogLevel = if ($Debug) { 'DEBUG' } else { 'WARNING' }

$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $BackgroundFile) {
    $BackgroundFile = Join-Path (Join-Path $BaseDir 'HYP_50M_SR_W') 'HYP_50M_SR_W.tif'
}
if (-not $GeographyFile) {
    $GeographyFile = Join-Path (Join-Path $BaseDir 'ne_50m_land') 'ne_50m_land.shp'
}

if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

$ModelFile = Join-Path $OutputDir 'dialect_classifier.bz2'
$PredictionFile = Join-Path $OutputDir 'predictions.csv'

Write-Host @"
Generate dialect maps
rule file = $RuleFile
annotation file = $AnnotationFile
output directory = $OutputDir
model file = $ModelFile
prediction file = $PredictionFile
"@

Write-Host "Training dialect classifier with rule file $RuleFile, annotation file $AnnotationFile..."
$ret = python -O (Join-Path $BaseDir 'dialect_classifier.py') `
    --log-level=$LogLevel `
    train `
    $RuleFile `
    $AnnotationFile `
    $ModelFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training dialect classifier failed: $ret"
    exit 1
}
Write-Host "Done. saved model to $ModelFile."

if ($ComplianceFile) {
    Write-Host "Using precomputed compliance file $ComplianceFile."
} else {
    $ComplianceFile = Join-Path $OutputDir 'compliances.csv'

    $ComplianceFiles = @()
    foreach ($dataset in $Datasets) {
        Write-Host "Computing rule compliances for dataset $dataset. This may take a while..."
        $cf = [System.IO.Path]::GetTempFileName()
        $ret = python -O -m sincomp.compare --rule-file=$RuleFile $dataset $cf
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Computing rule compliances failed: $ret"
            exit 1
        }
        $ComplianceFiles += $cf
        Write-Host 'Done.'
    }

    # merge compliance files, keeping header only from first
    $ComplianceFile = Join-Path $OutputDir 'compliances.csv'
    $first = $true
    foreach ($f in $ComplianceFiles) {
        $lines = Get-Content $f
        if ($first) {
            $lines | Set-Content $ComplianceFile
            $first = $false
        } else {
            $lines[1..($lines.Count-1)] | Add-Content $ComplianceFile
        }
    }
    Remove-Item $ComplianceFiles
}

Write-Host "Predicting dialects classes for $ComplianceFile..."
$ret = python -O (Join-Path $BaseDir 'dialect_classifier.py') `
    --log-level=$LogLevel `
    predict `
    --precomputed `
    --model=$ModelFile `
    $ComplianceFile `
    $PredictionFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Predicting dialect classes failed: $ret"
    exit 1
}
Write-Host "Done. saved predictions to $PredictionFile."

# download background and geography files if missing
if (($BackgroundFile -eq (Join-Path (Join-Path $BaseDir 'HYP_50M_SR_W') 'HYP_50M_SR_W.tif')) -and -not (Test-Path $BackgroundFile)) {
    Write-Host "Downloading $BackgroundFile from Natural Earth..."
    $tmp = Get-ChildItem ([System.IO.Path]::GetTempFileName()) |`
        Rename-Item -NewName { $_.FullName + '.zip' } -PassThru |`
        Select-Object -ExpandProperty FullName
    Invoke-WebRequest -Uri 'https://naciscdn.org/naturalearth/50m/raster/HYP_50M_SR_W.zip' -OutFile $tmp
    Expand-Archive -Force -Path $tmp -DestinationPath (Join-Path $BaseDir 'HYP_50M_SR_W')
    Remove-Item $tmp
    Write-Host 'Done.'
}

if (($GeographyFile -eq (Join-Path (Join-Path $BaseDir 'ne_50m_land') 'ne_50m_land.shp')) -and -not (Test-Path $GeographyFile)) {
    Write-Host "Downloading $GeographyFile from Natural Earth..."
    $tmp = Get-ChildItem ([System.IO.Path]::GetTempFileName()) |`
        Rename-Item -NewName { $_.FullName + '.zip' } -PassThru |`
        Select-Object -ExpandProperty FullName
    Invoke-WebRequest -Uri 'https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip' -OutFile $tmp
    Expand-Archive -Force -Path $tmp -DestinationPath (Join-Path $BaseDir 'ne_50m_land')
    Remove-Item $tmp
    Write-Host 'Done.'
}

Write-Host 'Generating dialect isoglosses for rule compliances. This may take a while...'

$extraOpts = @()
if ($BackgroundFile) {
    $extraOpts += "--background=$BackgroundFile"
}
if ($GeographyFile) {
    $extraOpts += "--geography=$GeographyFile"
}
if ($Extent) {
    $extraOpts += "--extent=$Extent"
}
if ($Size) {
    $extraOpts += "--size=$Size"
}

$ret = python -O (Join-Path $BaseDir 'isogloss.py') `
    --log-level=$LogLevel `
    --rule-file=$RuleFile `
    --output-prefix="$OutputDir\" `
    @extraOpts `
    $ComplianceFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Generating isoglosses failed: $ret"
    exit 1
}
Write-Host 'Done.'

Write-Host 'Generating dialect partition maps for predicted classes...'
$ret = python -O (Join-Path $BaseDir 'isogloss.py') `
    --log-level=$LogLevel `
    --output-prefix="$OutputDir\" `
    @extraOpts `
    $PredictionFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Generating dialect partition maps failed: $ret"
    exit 1
}
Write-Host 'Done.'

exit 0
