# Script PowerShell pentru a rula antrenamentul pe toate fold-urile MPHOI-72
# Utilizare: .\run_all_folds.ps1

$PYTHON_EXE = "c:\Users\Catalina\Desktop\Licenta\vhoip\venv\Scripts\python.exe"
$WORK_DIR = "c:\Users\Catalina\Desktop\Licenta\vhoip"
$NUM_FOLDS = 28

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Rulare antrenament MPHOI-72 pe 28 folduri" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

cd $WORK_DIR

$start_time = Get-Date
$total_folds = $NUM_FOLDS

for ($fold = 0; $fold -lt $NUM_FOLDS; $fold++) {
    Write-Host "[$fold/$($NUM_FOLDS - 1)] Incepe antrenament fold $fold..." -ForegroundColor Green
    Write-Host "Comanda: $PYTHON_EXE train.py --config configs/mphoi72.yaml --fold $fold --device cuda"
    Write-Host ""
    
    & $PYTHON_EXE train.py --config configs/mphoi72.yaml --fold $fold --device cuda
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Fold $fold completat cu succes!" -ForegroundColor Green
    } else {
        Write-Host "✗ Fold $fold a esuat cu exit code $LASTEXITCODE" -ForegroundColor Red
        Write-Host "Continuu cu urmatorul fold..." -ForegroundColor Yellow
    }
    Write-Host ""
}

$end_time = Get-Date
$total_duration = $end_time - $start_time

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Antrenament complet!" -ForegroundColor Cyan
Write-Host "Durata totala: $($total_duration.Hours)h $($total_duration.Minutes)m $($total_duration.Seconds)s" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Foldurile au fost rulate + logat in W&B cu numi:" -ForegroundColor Cyan
for ($fold = 0; $fold -lt $NUM_FOLDS; $fold++) {
    Write-Host "  - mphoi72_fold$fold" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Intra in W&B sa vezi toți metricile: https://wandb.ai/" -ForegroundColor Cyan
