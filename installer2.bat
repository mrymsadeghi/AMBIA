@echo off
:: Check if the file is provided as an argument
if "%~1"=="" (
    echo Usage: %~nx0 filename
    exit /b 1
)

:: Assign the first argument to the filename variable
set "filename=%~1"

:: Check if the file exists
if not exist "%filename%" (
    echo File not found: %filename%
    exit /b 1
)

:: Read the file line by line and install each package using pip
for /f "usebackq delims=" %%p in ("%filename%") do (
    :: Skip empty lines
    if not "%%p"=="" (
        echo Installing %%p...
        pip install %%p

        :: Check if the installation was successful
        if errorlevel 1 (
            echo Failed to install %%p
        ) else (
            echo Successfully installed %%p
        )
    )
)

echo All packages have been processed.