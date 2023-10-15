$scriptDirectory = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$pythonScript = Join-Path -Path $scriptDirectory -ChildPath "startFull.py"

# Replace venv path if required
$venvActivationScript = Join-Path -Path $scriptDirectory -ChildPath "..\myvenv\Scripts\Activate.ps1"

# Function to run the script
function RunScript {
    # Activate the virtual environment
    & $venvActivationScript

    # Run your Python script
    python $pythonScript
    # Deactivate the virtual environment
    deactivate
}

# Check and run the script at midnight
while ($true) {
    $currentTime = Get-Date
    if ($currentTime.Hour -eq 0 -and $currentTime.Minute -eq 0) {
        RunScript
        break  # Exit the loop after running the script
    }
    
    # Sleep for a while (e.g., 1 minute) before checking again
    Start-Sleep -Seconds 60
}
