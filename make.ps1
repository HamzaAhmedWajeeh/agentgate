<#
.SYNOPSIS
    Windows equivalent of the Makefile.

.DESCRIPTION
    GNU make is not present on a stock Windows install. This shim exposes the same target
    names so the documented commands work on every machine the project is developed on.
    Targets must stay in step with the Makefile; tests/unit/test_task_runner.py fails the
    build if one drifts.

.EXAMPLE
    ./make.ps1 check
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Step {
    param([Parameter(Mandatory)][string[]]$Command)
    Write-Host "> $($Command -join ' ')" -ForegroundColor DarkGray
    & $Command[0] @($Command[1..($Command.Length - 1)])
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$targets = [ordered]@{
    'setup'        = @{ Help = 'Install dependencies and git hooks'
        Steps = @(, @('uv', 'sync', '--all-groups'), @('uv', 'run', 'pre-commit', 'install')) }
    'lint'         = @{ Help = 'Check style and common defects'
        Steps = @(, @('uv', 'run', 'ruff', 'check', 'src', 'tests')) }
    'format'       = @{ Help = 'Rewrite files to the project format'
        Steps = @(, @('uv', 'run', 'ruff', 'format', 'src', 'tests')) }
    'format-check' = @{ Help = 'Fail if any file is unformatted'
        Steps = @(, @('uv', 'run', 'ruff', 'format', '--check', 'src', 'tests')) }
    'typecheck'    = @{ Help = 'Type check src in strict mode'
        Steps = @(, @('uv', 'run', 'mypy')) }
    'test'         = @{ Help = 'Run the offline suite (no API key required)'
        Steps = @(, @('uv', 'run', 'pytest')) }
    'test-cov'     = @{ Help = 'Run the offline suite with a coverage report'
        Steps = @(, @('uv', 'run', 'pytest', '--cov', '--cov-report=term-missing')) }
    'test-live'    = @{ Help = 'Run the suite against real providers. Estimates, confirms, then enforces.'
        Steps = @(, @('uv', 'run', 'python', 'scripts/run_live.py')) }
    'check'        = @{ Help = 'Everything CI runs'; DependsOn = @('lint', 'format-check', 'typecheck', 'test') }
    # Audits the exported lockfile, not the environment: agentgate itself is not on PyPI and
    # --strict treats an unauditable distribution as a failure.
    'audit'        = @{ Help = 'Check locked dependencies for known vulnerabilities'
        Steps = @(
            @('uv', 'export', '--all-groups', '--no-emit-project', '--no-hashes',
                '--format', 'requirements-txt', '-o', '.audit-requirements.txt', '--quiet'),
            @('uv', 'run', 'pip-audit', '--strict', '-r', '.audit-requirements.txt')) }
    'models'       = @{ Help = 'List model identifiers this key can reach, with a price-table skeleton'
        Steps = @(, @('uv', 'run', 'python', '-m', 'agentgate.models.catalogue')) }
    'config'       = @{ Help = 'Print the resolved configuration, secrets redacted'
        Steps = @(, @('uv', 'run', 'python', '-m', 'agentgate')) }
    'docker-build' = @{ Help = 'Build the container image'
        Steps = @(, @('docker', 'compose', 'build')) }
    'docker-up'    = @{ Help = 'Start the stack'
        Steps = @(, @('docker', 'compose', 'up', '-d')) }
    'docker-down'  = @{ Help = 'Stop the stack and remove volumes'
        Steps = @(, @('docker', 'compose', 'down', '-v')) }
    'docker-logs'  = @{ Help = 'Follow stack logs'
        Steps = @(, @('docker', 'compose', 'logs', '-f')) }
    'clean'        = @{ Help = 'Remove caches and build artefacts'; Steps = @() }
}

function Invoke-Target {
    param([Parameter(Mandatory)][string]$Name)

    if (-not $targets.Contains($Name)) {
        Write-Host "Unknown target '$Name'." -ForegroundColor Red
        Show-Help
        exit 2
    }

    $definition = $targets[$Name]
    foreach ($dependency in @($definition.DependsOn)) {
        if ($dependency) { Invoke-Target -Name $dependency }
    }
    foreach ($step in @($definition.Steps)) {
        if ($step) { Invoke-Step -Command $step }
    }
    if ($Name -eq 'clean') { Invoke-Clean }
}

function Invoke-Clean {
    $paths = @('.pytest_cache', '.mypy_cache', '.ruff_cache', '.coverage', 'htmlcov', 'dist', 'build')
    foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    }
    Get-ChildItem -Path . -Filter '__pycache__' -Recurse -Directory -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\\.venv\\' } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
}

function Show-Help {
    Write-Host ''
    foreach ($name in $targets.Keys) {
        Write-Host ('  {0,-14} {1}' -f $name, $targets[$name].Help)
    }
    Write-Host ''
}

if ($Target -in @('help', '-h', '--help')) { Show-Help; exit 0 }
Invoke-Target -Name $Target
