@echo off
set "PATH=%PATH%;C:\Users\HP\.mingit\cmd;C:\Users\HP\AppData\Local\Programs\Python\Python310\Scripts"
cd /d "%~dp0"
echo ========================================================
echo   MEDINSIGHT AI -- Pushing to GitHub Repository
echo ========================================================
echo Remote: https://github.com/vishwas-raizel/medical-price-.git
echo Branch: main
echo.
git push -u origin main
echo.
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo --------------------------------------------------------
    echo If prompted, please enter your GitHub Personal Access Token (PAT)
    echo or authorize in your browser.
    echo --------------------------------------------------------
)
pause
