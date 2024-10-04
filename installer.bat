@echo off
setlocal

:: Step 1: Create a new Python virtual environment and activate it
echo Creating Python virtual environment...
python -m venv ambia_env

:: Step 2: Activate the virtual environment
set "venvPath=%~dp0ambia_env\Scripts\activate"
call "%venvPath%"

:: Step 3: Upgrade pip to the latest version
echo Upgrading pip...
pip install --upgrade pip

:: Step 4: Install the required packages
echo Installing matplotlib...
pip install matplotlib > pip_log.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo There was an error installing matplotlib. Check pip_log.txt for details.
    exit /b 1
)

echo Installing packages from requirements.txt...
for /f "usebackq tokens=*" %%a in ("%~dp0requirements.txt") do (
    echo Installing %%a...
    pip install %%a >> pip_log.txt 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo There was an error installing %%a. Check pip_log.txt for details.
    )
)
echo Packages installed successfully

:: Step 5: Specify the subfolder path and file name
set "subfolderPath=mb_gui\src"
set "fileName=Main.py"
set "filePath=%~dp0%subfolderPath%\%fileName%"

:: Check if the file exists
if not exist "%filePath%" (
    echo File %filePath% not found.
    exit /b 1
)

echo File found at %filePath%.

:: Step 6: Create another batch file on the user's desktop
set "desktop=%USERPROFILE%\Desktop"
set "newBatFile=%desktop%\run_AMBIA.bat"

echo Creating batch file at %newBatFile%...
echo @echo off > "%newBatFile%"
echo call "%venvPath%" >> "%newBatFile%"
echo python "%filePath%" >> "%newBatFile%"

echo The batch file has been created on your desktop: %newBatFile%
pause
