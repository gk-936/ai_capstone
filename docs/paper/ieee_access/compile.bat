@echo off
REM Compiles paper.tex locally with the MiKTeX install in D:\overleaf.
REM Run this from inside docs\paper\ieee_access, or just double-click it.

REM A stray/malformed entry elsewhere in the system PATH (a file path listed
REM as if it were a directory, from a protoc install) makes MiKTeX abort on
REM startup if the full PATH is inherited. Using a minimal, known-good PATH
REM instead of prepending avoids that.
set PATH=C:\Windows\System32;C:\Windows;D:\overleaf\miktex\bin\x64

pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

echo.
echo Done. Output: paper.pdf
pause
