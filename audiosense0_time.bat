@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "SELF_DIR=%~dp0"

if "%~1"=="" (
  echo Usage:
  echo   audiosense0  "D:\path\to\file.wav"
  echo   audiosense0  "D:\path\to\folder"
  exit /b 1
)

set "TARGET=%~f1"
set "MODE="
if /i "%~x1"==".wav" if exist "%TARGET%" set "MODE=FILE"
if not defined MODE if exist "%TARGET%\*" set "MODE=DIR"
if not defined MODE (
  echo [ERROR] Not a .wav or existing folder: "%TARGET%"
  exit /b 2
)

REM ==========================================================
REM In DIR mode: write outputs into the folder TARGET
REM In FILE mode: write outputs into the folder of the file
REM ==========================================================

if /i "%MODE%"=="FILE" (
  call :PROCESS_ONE "%TARGET%"
  goto :END
)

echo [INFO] Folder mode: "%TARGET%"
for %%F in ("%TARGET%\*.wav") do (
  if exist "%%~fF" (
    call :PROCESS_ONE "%%~fF"
  )
)
goto :END


:PROCESS_ONE
setlocal EnableExtensions EnableDelayedExpansion
set "AUDIO=%~f1"
if not exist "!AUDIO!" (
  echo [WARN] Skipping missing: "!AUDIO!"
  endlocal & goto :EOF
)
set "STEM=%~n1"

REM Decide OUTDIR:
if /i "%MODE%"=="DIR" (
  set "OUTDIR=%TARGET%"
) else (
  set "OUTDIR=%~dp1"
)
REM Normalize OUTDIR
for %%Z in ("!OUTDIR!") do set "OUTDIR=%%~fZ"

echo.
echo ------------------------------------------------------------
echo [PROCESS] "!AUDIO!"
echo [OUTDIR ] "!OUTDIR!"
echo ------------------------------------------------------------

REM ==========================
REM Phase A: CLAP + argument in env "myenv"
REM ==========================
call conda activate myenv || goto :fail

echo [A1] CLAP (myenv) -> "!OUTDIR!\streaming_CLAP_!STEM!.csv"
call :Tic
python "%SELF_DIR%streaming_CLAP.py" "!AUDIO!" --out "!OUTDIR!\streaming_CLAP_!STEM!.csv" || goto :fail
call :Toc "CLAP Processing"

echo [A2] argument/pyannote (myenv) -> "!OUTDIR!\streaming_speakers_!STEM!.csv"
call :Tic
python "%SELF_DIR%streaming_argument.py" "!AUDIO!" --out "!OUTDIR!\streaming_speakers_!STEM!.csv" || goto :fail
call :Toc "Argument/Speakers"

set "SPKCSV=!OUTDIR!\streaming_speakers_!STEM!.csv"
set "SPKARG="
if exist "!SPKCSV!" (
  for %%A in ("!SPKCSV!") do if %%~zA gtr 0 set "SPKARG=--spk ""!SPKCSV!"""
)

REM ==========================
REM Phase B: rest in env "audioapp"
REM ==========================
call conda activate audioapp || goto :fail

echo [1/6] streaming AST transformer -> "!OUTDIR!\streaming_AST_!STEM!.csv"
call :Tic
python "%SELF_DIR%streaming_AST.py" "!AUDIO!" --out "!OUTDIR!\streaming_AST_!STEM!.csv" || goto :fail
call :Toc "AST Transformer"

echo [2/6] streaming ASR -> "!OUTDIR!\streaming_ASR_!STEM!.csv"
call :Tic
python "%SELF_DIR%streaming_ASR.py" --model large-v3 "!AUDIO!" --out "!OUTDIR!\streaming_ASR_!STEM!.csv" || goto :fail
call :Toc "ASR Transcription"

echo [3/6] pitch/prosody -> "!OUTDIR!\streaming_pitch_!STEM!.csv"
call :Tic
python "%SELF_DIR%streaming_pitch_online.py" "!AUDIO!" --out "!OUTDIR!\streaming_pitch_!STEM!.csv" || goto :fail
call :Toc "Pitch/Prosody"

REM echo [4/6] speaking rate (optional)
REM python "%SELF_DIR%streaming_speaking_rate.py" "!AUDIO!" --out "!OUTDIR!\streaming_prosody_!STEM!.csv" || goto :fail

echo [5/6] summarize -> "!OUTDIR!\summary_!STEM!.csv"
call :Tic
python "%SELF_DIR%summarize.py" ^
  --ast  "!OUTDIR!\streaming_AST_!STEM!.csv" ^
  --asr  "!OUTDIR!\streaming_ASR_!STEM!.csv" ^
  --clap "!OUTDIR!\streaming_CLAP_!STEM!.csv" ^
  --spk  "!OUTDIR!\streaming_speakers_!STEM!.csv" ^
  --pitch "!OUTDIR!\streaming_pitch_!STEM!.csv" ^
  --out  "!OUTDIR!\summary_!STEM!.csv" || goto :fail
call :Toc "Summarization"

echo [6/6] verdict (rolling) -> "!OUTDIR!\summary_with_comments_!STEM!_{0|1}.csv"
call :Tic
python "%SELF_DIR%verdict_rolling.py" --in "!OUTDIR!\summary_!STEM!.csv" --out "!OUTDIR!\summary_with_comments_!STEM!.csv" --prev-lines 4 || goto :fail
call :Toc "Verdict Rolling"

REM Find the actual suffixed verdict file
set "OUTCSV_BASE=!OUTDIR!\summary_with_comments_!STEM!"
set "CSV1=%OUTCSV_BASE%_1.csv"
set "CSV0=%OUTCSV_BASE%_0.csv"
set "CSVFOUND="
if exist "%CSV1%" set "CSVFOUND=%CSV1%"
if not defined CSVFOUND if exist "%CSV0%" set "CSVFOUND=%CSV0%"

if not defined CSVFOUND (
  echo [ERROR] Could not find verdict CSV: "%CSV1%" or "%CSV0%".
  goto :fail
)

echo [SRT] Building captions from "%CSVFOUND%"
pushd "!OUTDIR!" >nul
call :Tic
python "%SELF_DIR%csv_summary_to_srt.py" "%CSVFOUND%"
set "RC=%ERRORLEVEL%"
call :Toc "SRT Generation"
popd >nul
if not "%RC%"=="0" goto :fail

endlocal & goto :EOF


:fail
echo [ERROR] A step failed. Aborting this file.
endlocal & exit /b 1

:END
echo.
echo [DONE] audiosense0 finished.
endlocal & exit /b 0

REM ========================================================
REM TIMING HELPERS (Using Powershell for precision)
REM ========================================================
:Tic
REM Captures current time in seconds (Unix timestamp float)
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "Get-Date -UFormat %%s"`) do set "T_START=%%a"
exit /b

:Toc
REM Calculates elapsed time since :Tic
set "LABEL=%~1"
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "Get-Date -UFormat %%s"`) do set "T_END=%%a"
REM Perform subtraction in Powershell to handle decimals
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "%T_END% - %T_START%"`) do set "DIFF=%%a"
REM Format slightly to avoid excessive decimals if desired, or output raw
echo [TIME] %LABEL%: %DIFF% seconds
exit /b