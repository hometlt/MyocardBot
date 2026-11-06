<#
.SYNOPSIS
    Runs the checks and pushes the current commit to the server, which rebuilds
    and restarts the bot through the post-receive hook.

.EXAMPLE
    .\scripts\deploy.ps1
    .\scripts\deploy.ps1 -Server root@89.125.57.130 -SkipTests
#>
[CmdletBinding()]
param(
    # Only needed the first time, to create the "server" git remote.
    [string]$Server,
    [string]$RemoteName = "server",
    [string]$RepoPath = "/srv/miocard.git",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$remotes = git remote
if ($remotes -notcontains $RemoteName) {
    if (-not $Server) {
        throw "Remote '$RemoteName' is missing. Run once with -Server root@<server-ip>."
    }
    git remote add $RemoteName "ssh://$Server$RepoPath"
    if ($LASTEXITCODE -ne 0) { throw "Could not add the '$RemoteName' remote." }
    Write-Host "Added remote $RemoteName -> ssh://$Server$RepoPath"
}

if (git status --porcelain) {
    Write-Warning "The working tree has uncommitted changes; only the last commit is deployed."
}

if (-not $SkipTests) {
    Write-Host "==> ruff check"
    ruff check .
    if ($LASTEXITCODE -ne 0) { throw "ruff found problems; deployment stopped." }

    Write-Host "==> pytest"
    pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed; deployment stopped." }
}

$revision = (git rev-parse --short HEAD).Trim()
Write-Host "==> Pushing $revision to $RemoteName"
git push $RemoteName HEAD:master
if ($LASTEXITCODE -ne 0) { throw "Deployment failed; see the server output above." }

Write-Host "Deployed $revision" -ForegroundColor Green
