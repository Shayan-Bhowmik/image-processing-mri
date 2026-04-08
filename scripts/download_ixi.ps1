param(
    [string]$OutputDir = "data/raw/ixi"
)

$ErrorActionPreference = "Stop"

$downloads = @(
    @{ Name = "IXI-T1.tar"; Url = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar" },
    @{ Name = "IXI-T2.tar"; Url = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar" },
    @{ Name = "IXI-PD.tar"; Url = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar" },
    @{ Name = "IXI-MRA.tar"; Url = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-MRA.tar" }
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($item in $downloads) {
    $outFile = Join-Path $OutputDir $item.Name

    if (Test-Path $outFile) {
        $existing = Get-Item $outFile
        if ($existing.Length -gt 0) {
            Write-Host "Skipping existing file: $outFile ($([math]::Round($existing.Length / 1MB, 2)) MB)"
            continue
        }
    }

    Write-Host "Downloading $($item.Name)..."

    $success = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $item.Url -OutFile $outFile -UseBasicParsing
            $size = (Get-Item $outFile).Length
            if ($size -le 0) {
                throw "Downloaded file is empty"
            }
            Write-Host "Downloaded $($item.Name) ($([math]::Round($size / 1MB, 2)) MB)"
            $success = $true
            break
        }
        catch {
            Write-Warning "Attempt $attempt failed for $($item.Name): $($_.Exception.Message)"
            if ($attempt -eq 3) {
                throw
            }
            Start-Sleep -Seconds 2
        }
    }

    if (-not $success) {
        throw "Failed to download $($item.Name) after 3 attempts"
    }
}

Write-Host "All requested IXI archives are downloaded in: $OutputDir"
