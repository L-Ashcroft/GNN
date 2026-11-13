param([Parameter(Mandatory=$true)][string]$InputPath,
      [Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($InputPath, $false, $true, $false)
    $document.Repaginate()
    $document.ExportAsFixedFormat($OutputPath, 17)
    $document.Close(0)
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    $document = $null
    Write-Output 'Native Microsoft Word PDF export completed'
} finally {
    if ($null -ne $document) {
        $document.Close(0)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
}
