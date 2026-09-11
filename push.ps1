# One-click commit and push to GitHub (sayakwi3999-crypto/olist-recsys)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "== Repository: $PSScriptRoot ==" -ForegroundColor Cyan
Write-Host "== Changed files ==" -ForegroundColor Cyan
git status --short

$changes = git status --porcelain
if (-not $changes) {
    Write-Host "`nNothing to commit - the remote is already up to date." -ForegroundColor Yellow
    return
}

Write-Host ""
$msg = Read-Host "Commit message (press Enter for the default: Update <date>)"
if ([string]::IsNullOrWhiteSpace($msg)) {
    $msg = "Update " + (Get-Date -Format "yyyy-MM-dd")
}

Write-Host "`n[1/3] Staging changes ..." -ForegroundColor Cyan
git add -A

Write-Host "[2/3] Committing locally ..." -ForegroundColor Cyan
git commit -m $msg

Write-Host "[3/3] Pushing to GitHub ..." -ForegroundColor Cyan
git push

Write-Host "`nDone. View it at: https://github.com/sayakwi3999-crypto/olist-recsys" -ForegroundColor Green
