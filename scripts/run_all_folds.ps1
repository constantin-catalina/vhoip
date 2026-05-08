# Script PowerShell pentru a rula antrenamentul pe toate fold-urile unui dataset
# si a calcula automat FSUM mediu la final (ca in Tabelul 1 din paper).
# Utilizare:
#   .\scripts\run_all_folds.ps1                        # mphoi72, toate fold-urile
#   .\scripts\run_all_folds.ps1 -Dataset cad120        # cad120, toate fold-urile
#   .\scripts\run_all_folds.ps1 -Dataset mphoi72 -StartFold 2

param(
    [Parameter(Mandatory)]
    [ValidateSet("mphoi72", "cad120", "bimanual")]
    [string]$Dataset,
    [int]$StartFold = 0
)

$PYTHON_EXE = "c:\Users\Catalina\Desktop\Licenta\vhoip\venv\Scripts\python.exe"
$WORK_DIR = "c:\Users\Catalina\Desktop\Licenta\vhoip"

# Number of folds = C(num_subjects, 2): mphoi72=C(8,2)=28, cad120=C(4,2)=6, bimanual=C(6,2)=15
$NUM_FOLDS_MAP = @{ "mphoi72" = 28; "cad120" = 6; "bimanual" = 15 }
$CONFIG_MAP    = @{ "mphoi72" = "configs/mphoi72.yaml"; "cad120" = "configs/cad120.yaml"; "bimanual" = "configs/bimanual.yaml" }
$PAPER_FSUM    = @{ "mphoi72" = 188.6; "cad120" = 0.0; "bimanual" = 0.0 }

$NUM_FOLDS = $NUM_FOLDS_MAP[$Dataset]
$CONFIG    = $CONFIG_MAP[$Dataset]

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Rulare antrenament $Dataset pe $NUM_FOLDS folduri" -ForegroundColor Cyan
if ($StartFold -gt 0) {
    Write-Host "Incepand de la fold $StartFold (fold-urile 0-$($StartFold - 1) sarite)" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $WORK_DIR

$start_time = Get-Date

# Acumulator rezultate per fold
# Cheia = numarul fold-ului, valoarea = Best FSUM raportat de train.py
$fold_fsum = @{}   # fold -> FSUM
$failed_folds = @()   # fold-uri care au esuat

for ($fold = $StartFold; $fold -lt $NUM_FOLDS; $fold++) {

    Write-Host "----------------------------------------" -ForegroundColor DarkCyan
    Write-Host "[$($fold + 1)/$NUM_FOLDS] Fold $fold" -ForegroundColor Green
    Write-Host ""

    # Ruleaza train.py si captureaza intregul output in $output
    $output = & $PYTHON_EXE train.py `
        --config $CONFIG `
        --fold   $fold `
        --device cuda  2>&1

    # Afiseaza output-ul in timp real (il avem deja capturat, il printam)
    $output | Write-Host

    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Fold $fold a esuat (exit code $LASTEXITCODE)" -ForegroundColor Red
        $failed_folds += $fold
        continue
    }

    # Cauta linia: "Antrenare finalizata. Best FSUM: 213.4"
    # train.py scrie exact asta la sfarsit via logger.info()
    $fsum_line = $output | Where-Object { $_ -match "Best FSUM:\s*([\d.]+)" } | Select-Object -Last 1

    if ($fsum_line -match "Best FSUM:\s*([\d.]+)") {
        $fsum_val = [double]$Matches[1]
        $fold_fsum[$fold] = $fsum_val
        Write-Host "✓ Fold $fold completat — Best FSUM: $fsum_val" -ForegroundColor Green
    }
    else {
        Write-Host "✓ Fold $fold completat — FSUM negasit in output (verifica logurile)" -ForegroundColor Yellow
        $failed_folds += $fold
    }

    Write-Host ""
}

# -----------------------------------------------------------------------
# Sumar final
# -----------------------------------------------------------------------
$end_time = Get-Date
$total_duration = $end_time - $start_time
$completed_folds = $fold_fsum.Count

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ANTRENAMENT COMPLET" -ForegroundColor Cyan
Write-Host "Durata totala: $($total_duration.Hours)h $($total_duration.Minutes)m $($total_duration.Seconds)s" -ForegroundColor Cyan
Write-Host "Fold-uri completate: $completed_folds / $NUM_FOLDS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($failed_folds.Count -gt 0) {
    Write-Host "Fold-uri esuate / fara FSUM: $($failed_folds -join ', ')" -ForegroundColor Red
    Write-Host ""
}

if ($completed_folds -eq 0) {
    Write-Host "Niciun fold nu a returnat FSUM. Verifica logurile." -ForegroundColor Red
    exit 1
}

# Calculeaza media si std ale FSUM pe fold-urile completate
$fsum_values = $fold_fsum.Values | Sort-Object

$fsum_mean = ($fsum_values | Measure-Object -Average).Average
$fsum_min = ($fsum_values | Measure-Object -Minimum).Minimum
$fsum_max = ($fsum_values | Measure-Object -Maximum).Maximum

# Std deviation (PowerShell nu are Measure-Object -StdDev, calculam manual)
$sq_diffs = $fsum_values | ForEach-Object { [Math]::Pow($_ - $fsum_mean, 2) }
$variance = ($sq_diffs | Measure-Object -Average).Average
$fsum_std = [Math]::Sqrt($variance)

# Afiseaza tabel per fold
Write-Host "Rezultate per fold:" -ForegroundColor White
Write-Host "  Fold  |   FSUM" -ForegroundColor White
Write-Host "  ------|--------" -ForegroundColor DarkGray
foreach ($f in ($fold_fsum.Keys | Sort-Object)) {
    $val = $fold_fsum[$f]
    $mark = if ($val -eq $fsum_max) { " <-- best" } elseif ($val -eq $fsum_min) { " <-- worst" } else { "" }
    Write-Host ("  {0,4}  |  {1,6:F1}{2}" -f $f, $val, $mark) -ForegroundColor White
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ("  FSUM mediu  : {0:F1}" -f $fsum_mean) -ForegroundColor Yellow
Write-Host ("  FSUM std    : {0:F1}" -f $fsum_std) -ForegroundColor Yellow
Write-Host ("  FSUM min    : {0:F1}  (fold {1})" -f $fsum_min, ($fold_fsum.GetEnumerator() | Where-Object { $_.Value -eq $fsum_min } | Select-Object -First 1).Key) -ForegroundColor White
Write-Host ("  FSUM max    : {0:F1}  (fold {1})" -f $fsum_max, ($fold_fsum.GetEnumerator() | Where-Object { $_.Value -eq $fsum_max } | Select-Object -First 1).Key) -ForegroundColor White

$paper_ref = $PAPER_FSUM[$Dataset]
if ($paper_ref -gt 0) {
    $diff = $fsum_mean - $paper_ref
    $sign = if ($diff -ge 0) { "+" } else { "" }
    $color = if ($diff -ge 0) { "Green" } else { "Red" }
    Write-Host ("  vs paper    : {0}{1:F1} FSUM  (paper: {2:F1})" -f $sign, $diff, $paper_ref) -ForegroundColor $color
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "W&B dashboard: https://wandb.ai/" -ForegroundColor Cyan
