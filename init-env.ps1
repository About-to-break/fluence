param([switch]$Force)

Get-ChildItem -Recurse -File -Include '*.example','*.example.*' | ForEach-Object {
    $src = $_.FullName
    $name = $_.Name

    if ($name -like '*.example.env') {
        $dest = $src -replace '\.example\.env$', '.env'
    } elseif ($name -like '*.example.json') {
        $dest = $src -replace '\.example\.json$', '.json'
    } elseif ($name -match '\.example\..+') {
        $dest = $src -replace '\.example', ''
    } else {
        $dest = $src -replace '\.example$', '.env'
    }

    if (-not (Test-Path $dest) -or $Force) {
        Copy-Item -LiteralPath $src -Destination $dest -Force
        Write-Host "Copied $src -> $dest"
    } else {
        Write-Host "$dest already exists, skipping"
    }
}