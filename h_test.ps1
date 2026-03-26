# Save as: get_hardware_specs.ps1

Write-Host "=== GPU & VRAM SPECS ==="
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free,compute_cap --format=csv

Write-Host "`n=== MIG PARTITION CHECK ==="
nvidia-smi mig -lgip

Write-Host "`n=== CPU SPECS ==="
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors

Write-Host "`n=== SYSTEM RAM ==="
Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum | ForEach-Object { Write-Host "$([Math]::Round($_.Sum / 1GB, 2)) GB Total" }

# Run -> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process 
# Then -> .\h_test.ps1 | Tee-Object -FilePath res.txt