# RubAI Backend Server Startup Script
# This script ensures port 8000 is free before starting the server

Write-Host "🚀 Starting RubAI Backend Server..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill any processes using port 8000
Write-Host "🔍 Checking for processes using port 8000..." -ForegroundColor Yellow
$connections = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue

if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    Write-Host "⚠️  Found processes using port 8000: $($pids -join ', ')" -ForegroundColor Yellow
    
    foreach ($processId in $pids) {
        try {
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "   Killing process $processId ($($process.ProcessName))..." -ForegroundColor Red
                Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
                Start-Sleep -Milliseconds 500
            }
        }
        catch {
            Write-Host "   Could not kill process $processId (may already be dead)" -ForegroundColor Gray
        }
    }
    
    # Wait for Windows to clean up
    Write-Host "   Waiting for port to be released..." -ForegroundColor Gray
    Start-Sleep -Seconds 2
}
else {
    Write-Host "✅ Port 8000 is free" -ForegroundColor Green
}

# Step 2: Verify port is free
Write-Host ""
Write-Host "🔍 Final port check..." -ForegroundColor Yellow
$finalCheck = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue

if ($finalCheck) {
    Write-Host "❌ Port 8000 is still in use! Please try again or use a different port." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Port 8000 is ready!" -ForegroundColor Green
Write-Host ""

# Step 3: Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Step 4: Start the server
Write-Host ""
Write-Host "🌟 Starting Uvicorn server..." -ForegroundColor Cyan
Write-Host "   URL: http://localhost:8000" -ForegroundColor White
Write-Host "   Press CTRL+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Start uvicorn using venv python
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

