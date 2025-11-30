# Kill any process using port 8000
Write-Host "[INFO] Checking for processes using port 8000..." -ForegroundColor Cyan
Write-Host ""

# Method 1: Try Get-NetTCPConnection (PowerShell method)
$connections = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue

if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    Write-Host "[WARNING] Found $($pids.Count) process(es) using port 8000" -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($processId in $pids) {
        try {
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "   [KILL] Killing process $processId ($($process.ProcessName))..." -ForegroundColor Red
                Stop-Process -Id $processId -Force -ErrorAction Stop
                Start-Sleep -Milliseconds 500
                Write-Host "   [OK] Process $processId killed" -ForegroundColor Green
            } else {
                Write-Host "   [INFO] Process $processId no longer exists" -ForegroundColor Gray
            }
        }
        catch {
            Write-Host "   [WARNING] Could not kill process $processId - trying taskkill" -ForegroundColor Yellow
            # Try with taskkill as fallback
            try {
                $result = taskkill /F /PID $processId 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "   [OK] Process $processId killed with taskkill" -ForegroundColor Green
                }
            }
            catch {
                Write-Host "   [INFO] Process $processId may already be dead" -ForegroundColor Gray
            }
        }
    }
}

# Method 2: Use netstat as fallback (more reliable on Windows)
Write-Host ""
Write-Host "[INFO] Double-checking with netstat..." -ForegroundColor Cyan
$netstatOutput = netstat -ano | Select-String ":8000"

if ($netstatOutput) {
    foreach ($line in $netstatOutput) {
        if ($line -match '\s+(\d+)\s*$') {
            $procId = $Matches[1]
            Write-Host "   [KILL] Found PID $procId via netstat, killing..." -ForegroundColor Red
            try {
                taskkill /F /PID $procId 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "   [OK] Process $procId killed" -ForegroundColor Green
                } else {
                    Write-Host "   [INFO] Process $procId already gone or inaccessible" -ForegroundColor Gray
                }
            }
            catch {
                Write-Host "   [INFO] Process $procId may already be dead" -ForegroundColor Gray
            }
        }
    }
}

# Wait for Windows to release the port
Write-Host ""
Write-Host "[INFO] Waiting for port to be fully released..." -ForegroundColor Cyan
Start-Sleep -Seconds 3

# Final verification
$stillInUse = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
$netstatCheck = netstat -ano | Select-String ":8000"

if ($stillInUse -or $netstatCheck) {
    Write-Host ""
    Write-Host "[ERROR] Port 8000 is STILL in use!" -ForegroundColor Red
    Write-Host "[INFO] This may be a zombie process. Trying one more aggressive approach..." -ForegroundColor Yellow
    
    # Nuclear option: Kill all Python processes
    Write-Host "[WARNING] Killing ALL python.exe processes..." -ForegroundColor Yellow
    Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "   [KILL] Killing python.exe PID $($_.Id)..." -ForegroundColor Red
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    
    Start-Sleep -Seconds 2
    
    # Final check
    $finalCheck = netstat -ano | Select-String ":8000"
    if ($finalCheck) {
        Write-Host ""
        Write-Host "[ERROR] Port 8000 STILL in use after killing all Python processes!" -ForegroundColor Red
        Write-Host "[INFO] You may need to restart your computer or wait longer." -ForegroundColor Gray
        exit 1
    } else {
        Write-Host ""
        Write-Host "[OK] Port 8000 is now free!" -ForegroundColor Green
    }
}
else {
    Write-Host ""
    Write-Host "[OK] Port 8000 is completely free!" -ForegroundColor Green
}

Write-Host ""

